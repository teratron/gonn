---
name: gonn
description: >
  Generate high-performance GoNN neural network code following library best practices.
  Covers Builder and Options API styles, deep network construction, optimizer/scheduler
  configuration, and AI-Meta doc-comment annotations.
---

# GoNN Code Generation Skill

GoNN is a Go neural-network library with generics (`float32`/`float64`), two
construction styles, and a rich optimizer/regularizer/scheduler ecosystem.

## 1. API Style Selection

**Builder API (Style A)** — fluent chain; explicit per-layer control:

```go
n, err := nn.NewBuilder[float64]().
    Input(2).
    Dense(4, activation.SIGMOID, true).
    Output(1, activation.SIGMOID, true).
    WithLoss(loss.MSE).
    WithLearningRate(0.3).
    Compile()
```

**Options API (Style B)** — functional options; composable presets:

```go
n, err := nn.New[float64](
    nn.WithInput[float64](2),
    nn.WithHiddenLayer[float64](4, activation.SIGMOID),
    nn.WithOutput[float64](1, activation.SIGMOID),
    nn.WithLoss[float64](loss.MSE),
    nn.WithLearningRate[float64](0.3),
)
```

Rules:

- **Never mix styles** in one construction chain.
- Builder always ends with `.Compile()` or `.MustCompile()`.
- Options API compiles implicitly inside `nn.New` / `nn.MustNew`.

## 2. Construction Lifecycle

```
New → Configure (add layers, set hyperparams) → Compile → Train/Query
```

- `NewBuilder[T]()` — enters Configuring state.
- `Compile()` / `MustCompile()` — validates, builds graph, enters Operational.
- `nn.New[T](opts...)` — all three steps in one call.
- Builder/option calls **after** Compile are silent no-ops (logged at Warn).

## 3. Type Parameterization

Always specify the float type explicitly — never pass raw literals without context:

```go
// Correct — explicit type parameter
n, err := nn.New[float32](nn.WithInput[float32](4), ...)

// Wrong — misleading, T not constrained
// nn.New(nn.WithInput(4), ...)  ← does not compile
```

Use `float32` for memory-sensitive or embedded workloads; `float64` for precision.

## 4. Bulk Constructors for Deep Networks

For 10+ hidden layers use bulk constructors instead of repeated calls:

```go
// Uniform repeat (Builder)
n, _ := nn.NewBuilder[float64]().
    Input(784).
    Repeat(100, 256, activation.ReLU, true).   // 100 identical layers
    Output(10, activation.SOFTMAX, true).
    Compile()

// Pattern block (Options)
block := []nn.HiddenLayerSpec[float64]{
    {Size: 256, Activation: activation.ReLU, Bias: true},
    {Size: 128, Activation: activation.ReLU, Bias: true},
}
n, _ := nn.New[float64](
    nn.WithInput[float64](784),
    nn.Pattern[float64](block, 50),           // 100 layers (2×50)
    nn.WithOutput[float64](10, activation.SOFTMAX),
)

// Programmatic slice setter (Builder)
layers := make([]nn.HiddenLayerSpec[float64], 100)
for i := range layers {
    layers[i] = nn.HiddenLayerSpec[float64]{Size: 512, Activation: activation.ReLU, Bias: true}
}
n, _ := nn.NewBuilder[float64]().Input(784).HiddenLayers(layers).Output(10, activation.SOFTMAX, true).Compile()
```

## 5. Activation ↔ Loss Matching

| Output Activation | Recommended Loss | Task |
| :--- | :--- | :--- |
| `activation.SIGMOID` | `loss.BCE` | Binary classification (1 output) |
| `activation.SOFTMAX` | `loss.CCE` or `loss.CROSS_ENTROPY` | Multi-class classification |
| `activation.Linear` | `loss.MSE` or `loss.MAE` | Regression |
| `activation.TanH` | `loss.MSE` | Regression (bounded output) |

Common mistake: `SOFTMAX` + `MSE` compiles but converges poorly. GoNN logs a Warn.

## 6. Optimizer Selection and LR Scheduling

```go
// Simple baseline
opt := optimizer.NewSGD[float64](0.01)

// Deep networks — Adam recommended
opt := optimizer.NewAdam[float64](0.001)

// With LR scheduling (warm-up → cosine decay)
warmup := optimizer.NewWarmUpLR[float64](0.001, 1000)
cosine := optimizer.NewCosineAnnealingLR[float64](0.001, 1e-6, 9000)
chain := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
    {Scheduler: warmup, Duration: 1000},
    {Scheduler: cosine, Duration: 9000},
})
boundSched := optimizer.BindScheduler[float64](opt, chain)

n, _ := nn.New[float64](
    nn.WithOptimizer[float64](opt),
    nn.WithScheduler[float64](boundSched),
    // ... topology ...
)
```

`BindScheduler` connects the scheduler to the optimizer so `Step()` auto-updates
the optimizer's effective LR. Always call `BindScheduler` before passing to `WithScheduler`.

## 7. AI-Meta Annotation

Every exported identifier in GoNN carries an `AI-Meta:` trailing block:

```go
// MyFunction does X.
//
// AI-Meta:
//   - Purpose: One-line explanation of why this exists.
//   - Usage: Example call showing how to invoke it.
//   - Lifecycle: When it is valid to call (before/after Compile, etc.).
//   - Concurrency: Safe | ReadSafe | SingleGoroutine | NotSafe [; clarifier].
//   - Errors: Error sentinels this may return (e.g., ErrUserConfig).
//   - Related: [Symbol1], [Symbol2].
//   - Constraints: Invariants the caller must satisfy.
//   - Stability: Stable | Experimental | Deprecated | Internal.
func MyFunction() {}
```

Rules:

- Block appears after a blank doc-comment line.
- Maximum 12 lines including the `AI-Meta:` label.
- Never reference internal file paths or spec IDs.
- `Purpose` and `Stability` are always required for public API.

## 8. Error Handling

All GoNN errors wrap a category sentinel. Use `errors.Is` for routing:

```go
n, err := nn.New[float32](opts...)
if err != nil {
    switch {
    case errors.Is(err, utils.ErrUserConfig):
        log.Fatal("configuration mistake:", err)
    case errors.Is(err, utils.ErrInputData):
        log.Fatal("bad input data:", err)
    default:
        log.Fatal("unexpected error:", err)
    }
}
```

Available sentinels: `ErrUserConfig`, `ErrInputData`, `ErrCompute`, `ErrControl`,
`ErrIntegrity`, `ErrIO` — all in `pkg/utils`.

## 9. Testing Patterns

```go
// Table-driven test for training convergence
func TestXORConverges(t *testing.T) {
    n := nn.MustNew[float64](nn.PresetXOR[float64]())
    dataset := []nn.Sample[float64]{
        {Input: []float64{0, 0}, Target: []float64{0}},
        {Input: []float64{0, 1}, Target: []float64{1}},
        {Input: []float64{1, 0}, Target: []float64{1}},
        {Input: []float64{1, 1}, Target: []float64{0}},
    }
    _, loss, err := n.Fit(dataset)
    if err != nil {
        t.Fatal(err)
    }
    if loss > 0.01 {
        t.Errorf("loss %v > 0.01 after training", loss)
    }
}

// Benchmark for 100-layer Compile()
func BenchmarkDeepCompile(b *testing.B) {
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        _, _ = nn.NewBuilder[float32]().
            Input(784).
            Repeat(100, 256, activation.ReLU, true).
            Output(10, activation.SOFTMAX, true).
            Compile()
    }
}
```

Always run with `-race`: `go test -race ./...`

## 10. Common Mistakes

| Mistake | Fix |
| :--- | :--- |
| Calling `Train`/`Query` before `Compile` | Always call `Compile()` first; `New()` handles this automatically |
| Mixing Builder and Options in one chain | Use one style consistently per network |
| Passing `float64` literal where `T` is expected | Use `T(value)` or typed constant |
| Using `Sequential` and `Repeat` together | Pick one; `Repeat` adds explicit bias, `Sequential` uses `DefaultBias` |
| Omitting `BindScheduler` | Without it, `WithScheduler` advances the schedule but the optimizer LR never updates |
| `SOFTMAX` + `MSE` | Use `CCE` or `CROSS_ENTROPY` with SOFTMAX output |
| Zero-size hidden layer (`Dense(0, ...)`) | Compile returns `ErrUserConfig`; size must be > 0 |
