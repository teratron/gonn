# Options API — MNIST Classifier (Placeholder)

**Goal:** Construct an MNIST-scale classifier (784 → 128 → 64 → 10) using
the Options API (Style B) with the built-in preset and a custom optimizer.

## Code

```go
package main

import (
    "fmt"
    "log"

    "github.com/teratron/gonn/pkg/nn"
    "github.com/teratron/gonn/pkg/optimizer"
)

func main() {
    opt := optimizer.NewAdam[float32](0.001)

    // Style B: functional options — PresetMNIST bundles the full topology.
    net, err := nn.New[float32](
        nn.PresetMNIST[float32](),
        nn.WithOptimizer[float32](opt),
        nn.WithMaxIterations[float32](30),
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Network compiled: %d input, %d output\n",
        net.Config().InputSize, net.Config().OutputSize)
    // Further: load MNIST dataset via pkg/dataset and call net.Fit(samples).
}
// Output:
// Network compiled: 784 input, 10 output
```

## Explanation

1. `optimizer.NewAdam[float32](0.001)` — Adam adapts per-weight LR; good for MNIST.
2. `nn.PresetMNIST[float32]()` — wires 784→ReLU(128)→ReLU(64)→Softmax(10), CCE loss, He init.
3. `nn.WithOptimizer[float32](opt)` — overrides the default SGD optimizer.
4. `nn.WithMaxIterations[float32](30)` — caps training at 30 epochs.
5. `nn.New[float32](opts...)` — applies all options in order, then compiles.

## Anti-patterns

```go
// Wrong: using MustNew in production code that may fail
net := nn.MustNew[float32](badOpts...)  // panics — use New in production

// Wrong: calling WithHiddenLayer after PresetMNIST replaces topology
// PresetMNIST sets cfg.HiddenLayers directly — WithHiddenLayer appends to it.
// To fully replace topology, define your own option or use Builder API.
```
