---
name: go-struct-optimizer
description: Analyze and optimize Go struct memory layouts — padding, alignment, GC scan range. Use when reviewing struct field order or optimizing memory.
---

# Go Struct Optimizer

Analyze and optimize Go struct memory layouts: padding, alignment, GC scan range.

Logic inspired by [padiazg/go-struct-analyzer](https://github.com/padiazg/go-struct-analyzer)
and [fieldalignment](https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/fieldalignment).

## When to Use

- Reviewing or creating structs in performance-critical code
- Reducing memory footprint (padding elimination)
- Reducing GC pressure (pointer grouping)
- Before committing new struct definitions

## Quick Reference

### Type Sizes (amd64)

| Type | Size | Align | Pointer |
| --- | --- | --- | --- |
| `bool`, `int8`, `uint8`, `byte` | 1 | 1 | no |
| `int16`, `uint16` | 2 | 2 | no |
| `int32`, `uint32`, `rune`, `float32` | 4 | 4 | no |
| `int64`, `uint64`, `float64` | 8 | 8 | no |
| `complex64` | 8 | 4 | no |
| `complex128` | 16 | 8 | no |
| `int`, `uint`, `uintptr` | 8 | 8 | no |
| `*T`, `map[K]V`, `chan T`, `func(...)` | 8 | 8 | pure |
| `interface{}` / `any` | 16 | 8 | pure |
| `string` | 16 | 8 | mixed (ptr+len) |
| `[]T` | 24 | 8 | mixed (ptr+len+cap) |

### Pointer Classification

- **pure** — every word is a GC pointer (`*T`, `map`, `chan`, `func`, `interface{}`)
- **mixed** — first word is a pointer, rest are scalar (`string`, `[]T`)
- **none** — no pointer words (all numeric types, `bool`)

## Optimization Rules

### 1. Size-Optimal Order (minimize padding)

Sort fields by alignment descending, then by size descending, then by name:

```
alignment DESC → size DESC → name ASC
```

### 2. GC-Optimal Order (minimize scan range)

Sort fields by alignment descending, then by pointer class (pure → mixed → none),
with mixed sorted by size ascending (fewer trailing non-ptr words):

```
alignment DESC → ptr_class (pure < mixed < none) → size (ASC for mixed, DESC otherwise) → name ASC
```

### 3. Combined Strategy

For most cases, GC-optimal order also reduces padding. Prefer GC-optimal unless
struct has no pointers at all (then use size-optimal).

## Running the Script

The skill provides a Python analysis script at `scripts/analyze.py`.
No dependencies required — stdlib only.

```powershell
# Analyze a single file
python <skill_dir>/scripts/analyze.py <file.go>

# Analyze a directory tree
python <skill_dir>/scripts/analyze.py ./internal/ecs/

# Go-style recursive scan from the current dir (skips *_test.go, vendor,
# testdata, and dot/underscore dirs — same exclusions as `go ./...`)
python <skill_dir>/scripts/analyze.py ./...

# Recursive scan rooted at a subtree
python <skill_dir>/scripts/analyze.py ./internal/...
```

### Flags

| Flag | Effect |
| --- | --- |
| `--include-tests` | Also analyze `*_test.go` files. In Go, tests **and** benchmarks both live in `*_test.go`; there is no separate benchmark-file convention, so this one flag covers both. By default they are skipped because structs in test files are fixtures, not production hot-path data. |
| `--arch {amd64,arm64,386,arm}` | Target pointer width (default `amd64`). `amd64`/`arm64` → 8-byte pointers; `386`/`arm` → 4-byte. Note: `custom_types.json` sizes are authored for 64-bit. |

Unresolved named types (cross-package types not in `custom_types.json`) are
assumed `8B / align 8 / mixed` and listed on stderr so you can add precise
entries — ignore generic type parameters like `T`/`K`/`V` in that list.

### Output Format

For each struct the script prints:

```
=== StructName ===
  Current:  48 bytes (align 8), GC scan: 48 bytes
  Size-opt: 40 bytes (align 8), GC scan: 40 bytes  [saves 8B]
  GC-opt:   40 bytes (align 8), GC scan: 24 bytes  [scan -24B]

  Current layout:
    offset  size  pad  ptr   field
         0     8    0  pure  Entities    []Entity
         8     8    0  none  ID          uint64
        16    24    0  mix   Components  []ComponentID
        40     4    0  none  Count       uint32
        44     4    4  none  — padding —

  GC-optimal layout:
    offset  size  pad  ptr   field
         0     8    0  pure  Entities    []Entity
         8    24    0  mix   Components  []ComponentID
        32     8    0  none  ID          uint64
        40     4    0  none  Count       uint32
        44     4    4  none  — padding —
```

## Applying Fixes

After analysis, reorder struct fields manually following the GC-optimal layout.
Verify with:

```powershell
go vet ./...
go test -race ./...
```
