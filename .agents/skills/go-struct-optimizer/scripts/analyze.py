#!/usr/bin/env python3
"""Go struct memory layout analyzer.

Parses Go source files, calculates field sizes, alignment, padding, and GC scan
ranges, then suggests optimal field orderings.

Logic inspired by: https://github.com/padiazg/go-struct-analyzer (analyzer.ts)

Usage:
    python analyze.py <path> [path...] [--include-tests] [--arch ARCH]

Path forms:
    file.go            single file
    ./internal/ecs     a directory (non-recursive walk of that tree)
    ./...              Go-style recursive pattern from current dir
    ./internal/...     Go-style recursive pattern rooted at ./internal

By default *_test.go files (which hold both tests and benchmarks in Go) are
skipped; pass --include-tests to analyze them too.
"""

import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, field as dc_field
from typing import Optional

# --- Architecture / pointer width ---

ARCH_PTR_SIZE = {"amd64": 8, "arm64": 8, "386": 4, "arm": 4}
PTR_SIZE = 8  # set in main() from --arch

# Type table; rebuilt in main() once the architecture is known.
BASIC_TYPES: dict[str, tuple[int, int, str]] = {}

# Named types that could not be resolved (candidates for custom_types.json).
UNRESOLVED: set[str] = set()


def _load_custom_types() -> dict[str, tuple[int, int, str]]:
    custom: dict[str, tuple[int, int, str]] = {}
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "custom_types.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                custom[k] = (v["size"], v["align"], v["ptr"])
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 - report but keep going
        print(f"Warning: could not load custom_types.json: {e}", file=sys.stderr)
    return custom


def init_type_table(ptr_size: int) -> dict[str, tuple[int, int, str]]:
    """Build the (size, alignment, ptr_class) table for the given pointer width."""
    t: dict[str, tuple[int, int, str]] = {
        "bool": (1, 1, "none"),
        "int8": (1, 1, "none"),
        "uint8": (1, 1, "none"),
        "byte": (1, 1, "none"),
        "int16": (2, 2, "none"),
        "uint16": (2, 2, "none"),
        "int32": (4, 4, "none"),
        "uint32": (4, 4, "none"),
        "rune": (4, 4, "none"),
        "float32": (4, 4, "none"),
        "int64": (8, 8, "none"),
        "uint64": (8, 8, "none"),
        "float64": (8, 8, "none"),
        "complex64": (8, 4, "none"),
        "complex128": (16, 8, "none"),
        # Architecture-dependent widths.
        "int": (ptr_size, ptr_size, "none"),
        "uint": (ptr_size, ptr_size, "none"),
        "uintptr": (ptr_size, ptr_size, "none"),
        "string": (ptr_size * 2, ptr_size, "mixed"),
    }
    # Project-specific cross-package types override/extend the defaults.
    t.update(_load_custom_types())
    return t


@dataclass
class Field:
    name: str
    type_str: str
    size: int = 0
    alignment: int = 0
    ptr_class: str = "none"  # "pure" | "mixed" | "none"


@dataclass
class LayoutField:
    field: Field
    offset: int = 0
    padding: int = 0


@dataclass
class Layout:
    fields: list[LayoutField] = dc_field(default_factory=list)
    total_size: int = 0
    alignment: int = 0
    gc_scan: int = 0


@dataclass
class StructDef:
    name: str
    fields: list[Field] = dc_field(default_factory=list)
    line: int = 0
    multi_name: bool = False  # had `a, b T` declarations (expanded below)


# Registry for resolving embedded/named struct types within a file.
struct_registry: dict[str, list[Field]] = {}


# --- Type resolution ---


def resolve_type(
    type_str: str, visited: Optional[set[str]] = None
) -> tuple[int, int, str]:
    """Return (size, alignment, ptr_class) for a Go type string."""
    if visited is None:
        visited = set()

    clean = type_str.lstrip("*")

    # Pointer
    if type_str.startswith("*"):
        return PTR_SIZE, PTR_SIZE, "pure"

    # Slice []T
    if clean.startswith("[]"):
        return PTR_SIZE * 3, PTR_SIZE, "mixed"

    # Array [N]T
    arr_m = re.match(r"^\[(\d+)\](.+)$", clean)
    if arr_m:
        n = int(arr_m.group(1))
        elem_size, elem_align, elem_pc = resolve_type(arr_m.group(2), visited)
        pc = "none" if elem_pc == "none" else "mixed"
        return n * elem_size, elem_align, pc

    # Map
    if clean.startswith("map["):
        return PTR_SIZE, PTR_SIZE, "pure"

    # Channel
    if clean.startswith("chan ") or clean == "chan":
        return PTR_SIZE, PTR_SIZE, "pure"

    # Function
    if clean.startswith("func(") or clean.startswith("func "):
        return PTR_SIZE, PTR_SIZE, "pure"

    # Interface
    if clean in ("interface{}", "any") or clean.startswith("interface{"):
        return PTR_SIZE * 2, PTR_SIZE, "pure"

    # Basic / custom type
    if clean in BASIC_TYPES:
        return BASIC_TYPES[clean]

    # Strip package qualifier (e.g. "pkg.Type" -> "Type")
    base = clean.rsplit(".", 1)[-1] if "." in clean else clean

    # Registered struct
    if base not in visited and base in struct_registry:
        visited.add(base)
        fields = struct_registry[base]
        layout = compute_layout(fields)
        pc = "none"
        for f in fields:
            if f.ptr_class != "none":
                pc = "mixed"
                break
        return layout.total_size, layout.alignment, pc

    # Fallback: unknown named type — record it so the user can add a precise
    # entry to custom_types.json (the cleaned form matches the JSON key style).
    UNRESOLVED.add(clean)
    return PTR_SIZE, PTR_SIZE, "mixed"


# --- Layout computation ---


def calc_padding(offset: int, align: int) -> int:
    rem = offset % align
    return 0 if rem == 0 else align - rem


def compute_layout(fields: list[Field]) -> Layout:
    laid: list[LayoutField] = []
    offset = 0
    max_align = 1

    for f in fields:
        if f.alignment > max_align:
            max_align = f.alignment
        pad = calc_padding(offset, f.alignment)
        offset += pad
        laid.append(LayoutField(field=f, offset=offset, padding=pad))
        offset += f.size

    final_pad = calc_padding(offset, max_align)
    total = offset + final_pad

    # GC scan range: end offset of last pointer-containing field word.
    gc_scan = 0
    for lf in laid:
        if lf.field.ptr_class == "pure":
            end = lf.offset + lf.field.size
            if end > gc_scan:
                gc_scan = end
        elif lf.field.ptr_class == "mixed":
            end = lf.offset + PTR_SIZE  # only first word is a pointer
            if end > gc_scan:
                gc_scan = end

    return Layout(fields=laid, total_size=total, alignment=max_align, gc_scan=gc_scan)


# --- Sorting strategies ---


def size_optimal_order(fields: list[Field]) -> list[Field]:
    """Sort by alignment DESC, size DESC, name ASC."""
    return sorted(fields, key=lambda f: (-f.alignment, -f.size, f.name))


def gc_optimal_order(fields: list[Field]) -> list[Field]:
    """Sort: alignment DESC, ptr_class (pure<mixed<none), mixed size ASC, else size DESC, name ASC."""
    rank = {"pure": 0, "mixed": 1, "none": 2}

    def key(f: Field):
        r = rank[f.ptr_class]
        # Mixed: ascending size; pure/none: descending size
        size_key = f.size if f.ptr_class == "mixed" else -f.size
        return (-f.alignment, r, size_key, f.name)

    return sorted(fields, key=key)


# --- Parser ---


def extract_inline_comment(line: str) -> tuple[str, Optional[str]]:
    """Split a line into code part and optional // comment, respecting backtick tags."""
    in_backtick = False
    for i in range(len(line) - 1):
        if line[i] == "`":
            in_backtick = not in_backtick
        if not in_backtick and line[i] == "/" and line[i + 1] == "/":
            return line[:i].strip(), line[i:].strip()
    return line.strip(), None


def parse_field_line(line: str) -> list[Field]:
    """Parse a struct field line into one or more Fields.

    Multi-name declarations (`a, b, c T`) are expanded into one Field per
    name, so the layout math is exact — an improvement over the reference,
    which collapsed them to a single field.
    """
    code, _ = extract_inline_comment(line)

    # Strip struct tag `...`
    code = re.sub(r"\s*`[^`]+`\s*$", "", code).strip()
    if not code:
        return []

    # Embedded field: no whitespace (just a type like T, *T, pkg.T)
    if not re.search(r"\s", code):
        type_str = code
        name = type_str.lstrip("*").rsplit(".", 1)[-1]
        size, align, pc = resolve_type(type_str)
        return [
            Field(
                name=name,
                type_str=type_str,
                size=size,
                alignment=align,
                ptr_class=pc,
            )
        ]

    # Multi-name field: name1, name2 type  ->  one Field per name
    multi_m = re.match(r"^(\w+(?:\s*,\s*\w+)+)\s+(.+)$", code)
    if multi_m:
        names = [n.strip() for n in multi_m.group(1).split(",")]
        type_str = multi_m.group(2).strip()
        size, align, pc = resolve_type(type_str)
        return [
            Field(
                name=n,
                type_str=type_str,
                size=size,
                alignment=align,
                ptr_class=pc,
            )
            for n in names
        ]

    # Simple field: name type
    simple_m = re.match(r"^(\w+)\s+(.+)$", code)
    if simple_m:
        name = simple_m.group(1)
        type_str = simple_m.group(2).strip()
        size, align, pc = resolve_type(type_str)
        return [
            Field(
                name=name,
                type_str=type_str,
                size=size,
                alignment=align,
                ptr_class=pc,
            )
        ]

    return []


def parse_file(filepath: str) -> list[StructDef]:
    """Parse all struct definitions from a Go source file."""
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    structs: list[StructDef] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match: type Name[T any] struct {
        m = re.match(r"type\s+(\w+)(?:\[[^\]]*\])?\s+struct\s*\{?", line)
        if m:
            struct_name = m.group(1)
            start_line = i

            # Find opening brace
            if "{" not in line:
                i += 1
                while i < len(lines) and "{" not in lines[i]:
                    i += 1

            if i < len(lines):
                # Check for empty struct on same line
                brace_line = lines[i]
                net = brace_line.count("{") - brace_line.count("}")
                if net <= 0:
                    structs.append(StructDef(name=struct_name, line=start_line))
                    i += 1
                    continue

                # Parse fields
                i += 1
                fields: list[Field] = []
                multi_name = False
                brace_depth = 1
                while i < len(lines) and brace_depth > 0:
                    fl = lines[i].strip()
                    brace_depth += fl.count("{") - fl.count("}")
                    if brace_depth <= 0:
                        break
                    if fl and not fl.startswith("//") and not fl.startswith("/*"):
                        parsed = parse_field_line(fl)
                        if len(parsed) > 1:
                            multi_name = True
                        fields.extend(parsed)
                    i += 1

                # Register and store
                struct_registry[struct_name] = fields
                structs.append(
                    StructDef(
                        name=struct_name,
                        fields=fields,
                        line=start_line,
                        multi_name=multi_name,
                    )
                )
        i += 1

    # Second pass: re-resolve types now that all structs are registered
    for sd in structs:
        for f in sd.fields:
            f.size, f.alignment, f.ptr_class = resolve_type(f.type_str)

    return structs


# --- Output ---


def print_layout(label: str, layout: Layout):
    print(f"  {label}:")
    print(f"    {'offset':<7} {'size':<5} {'pad':<4} {'ptr':<5} {'field':<20} type")
    for lf in layout.fields:
        print(
            f"    {lf.offset:<7} {lf.field.size:<5} {lf.padding:<4} {lf.field.ptr_class:<5} "
            f"{lf.field.name:<20} {lf.field.type_str}"
        )
    # Tail padding
    if layout.fields:
        last = layout.fields[-1]
        tail_off = last.offset + last.field.size
        if tail_off < layout.total_size:
            gap = layout.total_size - tail_off
            print(f"    {tail_off:<7} {gap:<5} {'':4} {'':5} {'— tail padding —'}")
    print()


def _struct_can_optimize(sd: StructDef) -> bool:
    if not sd.fields:
        return False
    current = compute_layout(sd.fields)
    size_opt = compute_layout(size_optimal_order(sd.fields))
    gc_opt = compute_layout(gc_optimal_order(sd.fields))
    return (
        current.total_size - size_opt.total_size > 0
        or current.gc_scan - gc_opt.gc_scan > 0
    )


def print_report(sd: StructDef):
    if not sd.fields:
        return

    current = compute_layout(sd.fields)
    size_opt = compute_layout(size_optimal_order(sd.fields))
    gc_opt = compute_layout(gc_optimal_order(sd.fields))

    size_saved = current.total_size - size_opt.total_size
    gc_reduced = current.gc_scan - gc_opt.gc_scan

    if size_saved <= 0 and gc_reduced <= 0:
        return

    print(f"=== {sd.name} ===")
    if sd.multi_name:
        print(
            "  note: struct has grouped `a, b T` fields — expanded per name; "
            "preserve grouping when reordering manually"
        )
    print(
        f"  Current:  {current.total_size} bytes (align {current.alignment}), GC scan: {current.gc_scan} bytes"
    )

    if size_saved > 0:
        print(
            f"  Size-opt: {size_opt.total_size} bytes (align {size_opt.alignment}), "
            f"GC scan: {size_opt.gc_scan} bytes  [saves {size_saved}B]"
        )
    else:
        print(f"  Size-opt: {size_opt.total_size} bytes — already optimal")

    if gc_reduced > 0:
        print(
            f"  GC-opt:   {gc_opt.total_size} bytes (align {gc_opt.alignment}), "
            f"GC scan: {gc_opt.gc_scan} bytes  [scan -{gc_reduced}B]"
        )
    elif current.gc_scan > 0:
        print(f"  GC-opt:   GC scan {gc_opt.gc_scan} bytes — already optimal")
    print()

    print_layout("Current layout", current)
    if size_saved > 0 and gc_reduced <= 0:
        print_layout("Size-optimal layout", size_opt)
    if gc_reduced > 0:
        print_layout("GC-optimal layout", gc_opt)


def process_file(filepath: str):
    structs = parse_file(filepath)
    has_output = False
    for sd in structs:
        if _struct_can_optimize(sd):
            if not has_output:
                print(f"--- {filepath} ---\n")
                has_output = True
            print_report(sd)


# --- Path discovery ---

# Directories skipped during recursive (`./...`) walks — mirrors `go ./...`.
SKIP_DIRS = {"vendor", "testdata"}


def _is_target_go_file(fname: str, include_tests: bool) -> bool:
    if not fname.endswith(".go"):
        return False
    if not include_tests and fname.endswith("_test.go"):
        return False
    return True


def _walk(root: str, include_tests: bool):
    for cur, dirs, files in os.walk(root):
        # Prune ignored / hidden directories in place.
        dirs[:] = [
            d for d in dirs if d not in SKIP_DIRS and not d.startswith((".", "_"))
        ]
        for fname in sorted(files):
            if _is_target_go_file(fname, include_tests):
                process_file(os.path.join(cur, fname))


def process_path(path: str, include_tests: bool):
    # Go-style recursive pattern: "./...", "internal/...", "..."
    if path == "..." or path.endswith(("/...", "\\...")):
        root = path[:-3].rstrip("/\\") or "."
        if os.path.isdir(root):
            _walk(root, include_tests)
        else:
            print(f"Warning: not a directory: {root}", file=sys.stderr)
        return

    if os.path.isfile(path):
        if _is_target_go_file(os.path.basename(path), include_tests):
            process_file(path)
        return

    if os.path.isdir(path):
        _walk(path, include_tests)
        return

    print(f"Warning: path not found: {path}", file=sys.stderr)


def _force_utf8_streams():
    """Avoid mojibake for box-drawing dashes on legacy Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main():
    _force_utf8_streams()
    parser = argparse.ArgumentParser(
        description="Analyze Go struct memory layouts (padding, alignment, GC scan).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Path forms:\n"
            "  file.go            single file\n"
            "  ./internal/ecs     a directory tree\n"
            "  ./...              recursive from current dir\n"
            "  ./internal/...     recursive rooted at ./internal\n\n"
            "By default *_test.go files (tests AND benchmarks live there in Go)\n"
            "are skipped; use --include-tests to analyze them too."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="files, directories, or Go-style recursive patterns (e.g. ./...)",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="also analyze *_test.go files (tests and benchmarks)",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCH_PTR_SIZE),
        default="amd64",
        help="target architecture for pointer width (default: amd64)",
    )
    args = parser.parse_args()

    global PTR_SIZE, BASIC_TYPES
    PTR_SIZE = ARCH_PTR_SIZE[args.arch]
    BASIC_TYPES = init_type_table(PTR_SIZE)
    UNRESOLVED.clear()

    for arg in args.paths:
        process_path(arg, args.include_tests)

    if UNRESOLVED:
        print(
            "\nUnresolved named types (assumed 8B/align8/mixed). Add precise "
            "entries to custom_types.json for accuracy — ignore generic type "
            "parameters such as T/K/V:",
            file=sys.stderr,
        )
        for name in sorted(UNRESOLVED):
            print(f"  - {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
