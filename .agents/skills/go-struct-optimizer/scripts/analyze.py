#!/usr/bin/env python3
"""Go struct memory layout analyzer.

Parses Go source files, calculates field sizes, alignment, padding, and GC scan
ranges, then suggests optimal field orderings.

Logic inspired by: https://github.com/padiazg/go-struct-analyzer (analyzer.ts)

Usage:
    python analyze.py <file_or_dir> [file_or_dir...]
"""

import os
import re
import sys
import json
from dataclasses import dataclass, field as dc_field
from typing import Optional

PTR_SIZE = 8  # amd64

# --- Type size table (amd64) ---

BASIC_TYPES: dict[str, tuple[int, int, str]] = {
    # type: (size, alignment, ptr_class)
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
    "int": (8, 8, "none"),
    "uint": (8, 8, "none"),
    "uintptr": (8, 8, "none"),
    "string": (PTR_SIZE * 2, PTR_SIZE, "mixed"),
}


def _load_custom_types() -> dict[str, tuple[int, int, str]]:
    custom = {}
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "custom_types.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                custom[k] = (v["size"], v["align"], v["ptr"])
    except Exception as e:
        print(f"Warning: could not load custom_types.json: {e}", file=sys.stderr)
    return custom


BASIC_TYPES.update(_load_custom_types())


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

    # Basic type
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

    # Fallback: unknown named type
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

    # GC scan range: end offset of last pointer-containing field word
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


def parse_field_line(line: str) -> Optional[Field]:
    """Parse a single struct field line into a Field."""
    code, _ = extract_inline_comment(line)

    # Strip struct tag `...`
    code = re.sub(r"\s*`[^`]+`\s*$", "", code).strip()
    if not code:
        return None

    # Embedded field: no whitespace (just a type like T, *T, pkg.T)
    if not re.search(r"\s", code):
        type_str = code
        name = type_str.lstrip("*").rsplit(".", 1)[-1]
        size, align, pc = resolve_type(type_str)
        return Field(
            name=name, type_str=type_str, size=size, alignment=align, ptr_class=pc
        )

    # Multi-name field: name1, name2 type  →  treat each as separate field
    multi_m = re.match(r"^(\w+(?:\s*,\s*\w+)+)\s+(.+)$", code)
    if multi_m:
        # Take first name only (simplification)
        name = multi_m.group(1).split(",")[0].strip()
        type_str = multi_m.group(2).strip()
        size, align, pc = resolve_type(type_str)
        return Field(
            name=name, type_str=type_str, size=size, alignment=align, ptr_class=pc
        )

    # Simple field: name type
    simple_m = re.match(r"^(\w+)\s+(.+)$", code)
    if simple_m:
        name = simple_m.group(1)
        type_str = simple_m.group(2).strip()
        size, align, pc = resolve_type(type_str)
        return Field(
            name=name, type_str=type_str, size=size, alignment=align, ptr_class=pc
        )

    return None


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
                brace_depth = 1
                while i < len(lines) and brace_depth > 0:
                    fl = lines[i].strip()
                    brace_depth += fl.count("{") - fl.count("}")
                    if brace_depth <= 0:
                        break
                    if fl and not fl.startswith("//") and not fl.startswith("/*"):
                        field = parse_field_line(fl)
                        if field:
                            fields.append(field)
                    i += 1

                # Register and store
                struct_registry[struct_name] = fields
                structs.append(
                    StructDef(name=struct_name, fields=fields, line=start_line)
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
        if not sd.fields:
            continue
        current = compute_layout(sd.fields)
        size_opt = compute_layout(size_optimal_order(sd.fields))
        gc_opt = compute_layout(gc_optimal_order(sd.fields))
        if (
            current.total_size - size_opt.total_size > 0
            or current.gc_scan - gc_opt.gc_scan > 0
        ):
            if not has_output:
                print(f"--- {filepath} ---\n")
                has_output = True
            print_report(sd)


def process_path(path: str):
    if os.path.isfile(path):
        if path.endswith(".go") and not path.endswith("_test.go"):
            process_file(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if fname.endswith(".go") and not fname.endswith("_test.go"):
                    process_file(os.path.join(root, fname))


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze.py <file_or_dir> [file_or_dir...]", file=sys.stderr
        )
        sys.exit(1)

    for arg in sys.argv[1:]:
        process_path(arg)


if __name__ == "__main__":
    main()
