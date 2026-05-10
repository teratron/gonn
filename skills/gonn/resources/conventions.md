# GoNN Coding Conventions

Guidelines for AI agents generating or modifying GoNN library code.

## Generic Float Type Constraint

All numeric computation uses `utils.Float` (`float32 | float64`). Never hardcode `float64`:

```go
// Correct
func MyFunc[T utils.Float](x T) T { return x * 2 }

// Wrong
func MyFunc(x float64) float64 { return x * 2 }
```

Import: `"github.com/teratron/gonn/pkg/utils"`

## Compile-Time Interface Verification

Every concrete type implementing an interface must include a compile-time check:

```go
// In the same file as the type definition, after the type declaration:
var _ optimizer.Optimizer[float32] = (*MyOptimizer[float32])(nil)
```

This prevents interface drift — missing methods are caught at compile time, not runtime.

## Dispatcher Pattern (Activations and Losses)

Enum-based types follow a strict 3-part pattern:

1. `Type uint8` with a `String()` method.
2. One implementation file per enum value.
3. A switch-based dispatcher function.

Adding a new activation or loss requires all three steps. Never add a case without
the corresponding implementation file.

## Composition via Embedding

Type hierarchies use Go struct embedding, not interface embedding:

```go
// Correct: base → extended → exported
type base[T utils.Float] struct { ... }
type extended[T utils.Float] struct { base[T]; extra field }
type Exported[T utils.Float] struct { extended[T]; public field }
```

## Zero External Dependencies

Only Go standard library is permitted. No third-party imports — including `testify`.
Use `testing.T` methods (`t.Errorf`, `t.Fatalf`, `t.Run`) directly.

## Test Coverage

Every new package must reach 80% line coverage:

```bash
go test -cover ./pkg/mypackage/...
```

Required test types:
- `Test*` functions with `t.Run` subtests for ≥3 scenarios.
- `Benchmark*` functions for hot paths (allocation target: 0 allocs/op after warm-up).
- Always run with `-race`: `go test -race ./...`

## Doc-Comment Verbosity

Public API (`pkg/nn/`, `cmd/`) requires full doc comment:

```go
// FunctionName does X because Y.
// Parameters: param is Z.
// Returns: result is Q on success, ErrUserConfig if ...
//
// Example:
//   result := FunctionName(arg)
//
// AI-Meta:
//   - Purpose: One-line reason this exists.
//   - Usage: Example call.
//   - Lifecycle: When valid to call.
//   - Concurrency: Safe | ReadSafe | SingleGoroutine | NotSafe.
//   - Errors: Sentinel types this may return.
//   - Related: [Symbol1], [Symbol2].
//   - Stability: Stable.
func FunctionName(param Type) (ResultType, error) { ... }
```

First sentence starts with the identifier name: `// FunctionName does ...`

## Error Informativeness

Errors must be specific and actionable:

```go
// Correct
return utils.Newf(utils.ErrUserConfig,
    "Input(): size must be positive, got %d", size)

// Wrong
return errors.New("invalid input")  // too vague
```

Every error wraps a category sentinel from `pkg/utils/errors.go`:
`ErrUserConfig`, `ErrInputData`, `ErrCompute`, `ErrControl`, `ErrIntegrity`, `ErrIO`.

Use `fmt.Errorf("...: %w", err)` to preserve the chain.

## AI-Meta Annotation Closed Vocabulary

Fields allowed in the `AI-Meta:` block (no others):

| Field | Required for public API |
|:---|:---|
| `Purpose` | Always |
| `Usage` | Recommended |
| `Lifecycle` | If lifecycle matters |
| `Concurrency` | If concurrent use is relevant |
| `Errors` | If the function returns errors |
| `Related` | Recommended |
| `Constraints` | If invariants must be respected |
| `Implementations` | For interfaces only |
| `Stability` | Always |

`Stability` enum: `Stable` / `Experimental` / `Deprecated` / `Internal`
`Concurrency` enum: `Safe` / `ReadSafe` / `SingleGoroutine` / `NotSafe`

Hard cap: ≤ 12 lines including the `AI-Meta:` label.
Never reference internal file paths, spec filenames, or rule numbers.
