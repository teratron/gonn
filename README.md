# GoNN

[![GoDoc](https://godoc.org/github.com/teratron/gonn?status.svg)](https://godoc.org/github.com/teratron/gonn)
[![Go Report Card](https://goreportcard.com/badge/github.com/teratron/gonn)](https://goreportcard.com/report/github.com/teratron/gonn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

GoNN is a generic, zero-dependency Go library for building and training feedforward neural networks. It supports multi-hidden topologies, pluggable optimizers (SGD, Adam, RMSProp, SGD+Momentum), and regularization (L1, L2, Dropout, Compose). The public API uses Go generics (`float32` / `float64`).

## Installation

```shell
go get github.com/teratron/gonn
```

## Quick Start

### Functional Options API

```go
import (
    "github.com/teratron/gonn/pkg/activation"
    "github.com/teratron/gonn/pkg/nn"
    "github.com/teratron/gonn/pkg/optimizer"
)

n, err := nn.New[float32](
    nn.WithInput[float32](2),
    nn.WithHiddenLayer[float32](4, activation.SIGMOID),
    nn.WithOutput[float32](1, activation.SIGMOID),
    nn.WithLearningRate[float32](0.3),
)

// Train
loss, err := n.Train([]float32{0, 1}, []float32{1})

// Inference
out, err := n.Query([]float32{0, 1})
```

### Builder API

```go
n, err := nn.NewBuilder[float64]().
    Input(2).
    Dense(4, activation.SIGMOID, true).
    Dense(4, activation.SIGMOID, true).
    Output(1, activation.SIGMOID, true).
    WithLearningRate(0.01).
    Compile()
```

## v0.6 Features

### Multi-Hidden Topology

Arbitrary hidden-layer chains are supported. Mix activations and bias settings per layer:

```go
n, err := nn.New[float32](
    nn.WithInput[float32](4),
    nn.WithHiddenLayer[float32](16, activation.ReLU),
    nn.WithHiddenLayer[float32](8,  activation.ReLU),
    nn.WithOutput[float32](3, activation.SOFTMAX),
    nn.WithWeightInit[float32](nn.WeightInitHe),
)
```

### Pluggable Optimizers

```go
import "github.com/teratron/gonn/pkg/optimizer"

// Adam
n, _ := nn.New[float32](
    nn.WithInput[float32](4),
    nn.WithHiddenLayer[float32](8, activation.SIGMOID),
    nn.WithOutput[float32](1, activation.SIGMOID),
    nn.WithOptimizer[float32](optimizer.NewAdam[float32](0.001)),
)

// SGD with momentum
nn.WithOptimizer[float32](optimizer.NewSGDMomentum[float32](0.01))

// RMSProp
nn.WithOptimizer[float32](optimizer.NewRMSProp[float32](0.001))
```

### Regularization

```go
import "github.com/teratron/gonn/pkg/regularizer"

// L2 weight decay
n, _ := nn.New[float32](
    nn.WithInput[float32](4),
    nn.WithHiddenLayer[float32](8, activation.SIGMOID),
    nn.WithOutput[float32](1, activation.SIGMOID),
    nn.WithRegularizer[float32](regularizer.NewL2[float32](0.01)),
)

// Compose L2 + Dropout
reg := regularizer.Compose[float32](
    regularizer.NewL2[float32](0.01),
    regularizer.NewDropout[float32](0.8),
)
nn.WithRegularizer[float32](reg)
```

### Weight Initialization

Three strategies available via `WithWeightInit`:

| Constant | Formula | Recommended for |
| --- | --- | --- |
| `WeightInitXavier` (default) | Glorot uniform U[-a, a] | SIGMOID / TanH |
| `WeightInitHe` | He normal N(0, 2/fanIn) | ReLU / LeakyReLU |
| `WeightInitRandom` | Uniform U[-1, 1) | Shallow / experimental |

## Managed Training Loop

`Fit` runs the multi-epoch loop with early stopping, best-weight rollback, and optional callbacks:

```go
dataset := []nn.Sample[float32]{
    {Input: []float32{0, 0}, Target: []float32{0}},
    {Input: []float32{0, 1}, Target: []float32{1}},
    {Input: []float32{1, 0}, Target: []float32{1}},
    {Input: []float32{1, 1}, Target: []float32{0}},
}

epochs, loss, err := n.Fit(dataset)
```

Callbacks for monitoring:

```go
nn.WithEpochCallback[float32](func(epoch uint, loss float32) {
    fmt.Printf("epoch %d  loss=%.6f\n", epoch, loss)
})
```

Concurrency controls (safe to call from another goroutine while Fit is running):

```go
n.Pause()   // block before next epoch
n.Resume()  // unblock
n.Stop()    // terminate loop, return completed epochs
```

## Example Catalog

| ID | Directory | Description | API used |
| --- | --- | --- | --- |
| E01 | `examples/xor/` | XOR — dual-style (Builder + Options) | Both |
| E02 | `examples/logic_gates/` | AND / OR / NAND truth tables | Options |
| E03 | `examples/perceptron/` | 4-hidden SIGMOID perceptron | Builder |
| E04 | `examples/binary_classification/` | 2-hidden BCE + He init, Gaussian blobs | Options |
| E05 | `examples/iris/` | 3-class Iris CSV, SoftMax, EpochCallback | Options |
| E07 | `examples/regression_sin/` | Sine regression, TanH, RMSE ≤ 0.10 | Builder |
| E08 | `examples/regression_multi/` | 5-input 3-output ReLU + He | Options |
| E09 | `examples/persistence/` | Save / reload weights via `pkg/persistence` | Options |
| E11 | `examples/callbacks/` | EpochCallback + BatchCallback | Options |
| E12 | `examples/style_showcase/` | Same XOR built three ways | Both |
| E13 | `examples/higher_order_options/` | Sequential + DeepNetwork preset | Options |
| E14 | `examples/shared_options/` | Shared `[]Option[T]` across topologies | Options |
| E15 | `examples/precision/` | float32 vs float64 comparison | Both |
| E06 | _(deferred)_ | MNIST — awaits dataset-loader spec | — |
| E10 | _(deferred)_ | Continuation training — awaits `AndTrain` API | — |

## Package Overview

| Package | Purpose |
| --- | --- |
| `pkg/nn` | Public facade — `NN[T]`, Builder API, Options API, `Fit`, `Train`, `Query` |
| `pkg/optimizer` | Weight-update strategies: SGD, Adam, RMSProp, SGDMomentum |
| `pkg/regularizer` | Regularization: L1, L2, Dropout, Compose |
| `pkg/network` | Multi-hidden network graph, forward/backward propagation |
| `pkg/layer` | Input, Dense, Output layer constructors |
| `pkg/neuron` | Cell and axon primitives |
| `pkg/activation` | Activation functions and derivatives |
| `pkg/loss` | Loss functions (MSE, BCE, CCE, MAE, Huber, …) |
| `pkg/persistence` | JSON config + weight persistence (schema 1.1.0) |
| `pkg/checkpoint` | Resumable training snapshots with gzip cold tier |
| `pkg/dataset` | Slice, CSV, and prefetch dataset sources |
| `pkg/compute` | Pluggable backend interface + CPU reference backend |

## License

[MIT License](LICENSE).
