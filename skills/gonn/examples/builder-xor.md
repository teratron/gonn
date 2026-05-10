# Builder API — XOR Classifier

**Goal:** Train a 2-4-1 network with SIGMOID activation to solve XOR using
the Builder API (Style A).

## Code

```go
package main

import (
    "fmt"
    "log"

    "github.com/teratron/gonn/pkg/activation"
    "github.com/teratron/gonn/pkg/loss"
    "github.com/teratron/gonn/pkg/nn"
)

func main() {
    // Style A: fluent builder chain.
    net, err := nn.NewBuilder[float64]().
        Input(2).
        Dense(4, activation.SIGMOID, true).
        Output(1, activation.SIGMOID, true).
        WithLoss(loss.MSE).
        WithLearningRate(0.3).
        WithMaxIterations(5000).
        Compile()
    if err != nil {
        log.Fatal(err)
    }

    dataset := []nn.Sample[float64]{
        {Input: []float64{0, 0}, Target: []float64{0}},
        {Input: []float64{0, 1}, Target: []float64{1}},
        {Input: []float64{1, 0}, Target: []float64{1}},
        {Input: []float64{1, 1}, Target: []float64{0}},
    }

    epochs, finalLoss, err := net.Fit(dataset)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Trained %d epochs, loss=%.6f\n", epochs, finalLoss)

    // Inference
    out, _ := net.Query([]float64{0, 1})
    fmt.Printf("XOR(0,1) = %.4f (expected ~1)\n", out[0])
}
// Output:
// Trained ... epochs, loss=...
// XOR(0,1) = 0.9... (expected ~1)
```

## Explanation

1. `NewBuilder[float64]()` — enters Configuring state with `float64` precision.
2. `.Input(2)` — declares 2 input features.
3. `.Dense(4, activation.SIGMOID, true)` — one hidden layer: 4 neurons, SIGMOID, bias on.
4. `.Output(1, activation.SIGMOID, true)` — 1 output neuron, SIGMOID for binary output.
5. `.WithLoss(loss.MSE)` — mean-squared error is standard for SIGMOID regression.
6. `.Compile()` — validates, builds graph, transitions to Operational.
7. `net.Fit(dataset)` — runs the managed training loop with best-weight rollback.

## Anti-patterns

```go
// Wrong: calling Train before Compile
n := nn.NewBuilder[float64]()
n.Train(input, target)  // panics or returns ErrUserConfig

// Wrong: mixing Builder and Options
n := nn.NewBuilder[float64]().Input(2)
nn.WithHiddenLayer[float64](4, activation.SIGMOID)(&n.cfg)  // not the Builder API

// Wrong: SIGMOID output with CCE loss
.Output(1, activation.SIGMOID, true).WithLoss(loss.CCE)  // use BCE for single-output binary
```
