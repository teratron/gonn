# GoNN

[![GoDoc](https://godoc.org/github.com/teratron/gonn?status.svg)](https://godoc.org/github.com/teratron/gonn)
[![Go Report Card](https://goreportcard.com/badge/github.com/teratron/gonn)](https://goreportcard.com/report/github.com/teratron/gonn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

GoNN is a generic, zero-dependency Go library for building and training feedforward neural networks. It supports multi-hidden topologies, pluggable optimizers (SGD, Adam, RMSProp, SGD+Momentum), regularization (L1, L2, honest per-layer Dropout, Compose), normalization layers (BatchNorm, LayerNorm, GroupNorm), LR schedulers, conv/recurrent/attention prefix layers, persistence (`Save`/`Load`), training checkpoints, streaming datasets, and an HTTP observability endpoint. The public API uses Go generics (`float32` / `float64`).

Training numerics are guarded by a numeric gradient-check suite (`internal/verification/`): analytic gradients are validated against central differences across topologies, activations, and normalization layers on every CI run.

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
n.Pause()   // park the training goroutine before the next epoch (zero CPU)
n.Resume()  // unblock
n.Stop()    // terminate loop, return completed epochs; the network stays usable
```

Concurrent inference is race-free: any number of goroutines may `Query` a
compiled dense network in parallel (stateless forward under a read lock).

## Persistence, Checkpoints, Streaming

```go
// Save / reload a trained network (config + weights, hash-linked).
err := n.Save("config.json", "weights.json")
n2, err := nn.Load[float32]("config.json", "weights.json")

// Periodic training snapshots with retention, and resume:
n, _ := nn.New[float32](
    /* topology... */
    nn.WithCheckpoint[float32]("ckpts", 10, checkpoint.SweepConfig{}),
)
resumed, epoch, err := nn.Resume[float32]("ckpts")

// Streaming training over a dataset source (epoch = drain + Reset):
ds, _ := dataset.NewCSVDataset[float32]("train.csv", inSize, outSize, batchSize)
epochs, loss, err := n.FitDataset(ctx, ds)
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
| E09 | `examples/persistence/` | Save / reload round-trip via `nn.Save` / `nn.Load` | Builder |
| E10 | `examples/continuation/` | Continuation training — `AndTrain` parity | Options |
| E11 | `examples/callbacks/` | EpochCallback + BatchCallback | Options |
| E12 | `examples/style_showcase/` | Same XOR built three ways | Both |
| E13 | `examples/higher_order_options/` | Sequential + DeepNetwork preset | Options |
| E14 | `examples/shared_options/` | Shared `[]Option[T]` across topologies | Options |
| E15 | `examples/precision/` | float32 vs float64 comparison | Both |
| E06 | `examples/mnist/` | MNIST dense classifier (IDX loader, `-download` flag) | Options |
| E16 | `examples/mnist_cnn/` | MNIST with a Conv2D prefix stack | Options |


## Package Overview

| Package | Purpose |
| --- | --- |
| `pkg/nn` | Public facade — `NN[T]`, Builder API, Options API, `Fit`, `FitDataset`, `Train`, `Query`, `Save`/`Load`, `Resume` |
| `pkg/optimizer` | Weight-update strategies (SGD, Adam, RMSProp, SGDMomentum) + LR schedulers |
| `pkg/regularizer` | Regularization: L1, L2, per-layer Dropout, Compose |
| `pkg/network` | Multi-hidden network graph, forward/backward propagation, norm/mask hooks |
| `pkg/layer` | Input, Dense, Output layer constructors |
| `pkg/layer/conv` | Conv1D/Conv2D, pooling, flatten prefix layers |
| `pkg/layer/recurrent` | SimpleRNN, LSTM, GRU, LastStep prefix layers |
| `pkg/layer/attention` | Multi-head attention |
| `pkg/layer/transformer` | Encoder/Decoder blocks and stacks |
| `pkg/layer/embedding` | Token embedding + positional encoding |
| `pkg/layer/norm` | BatchNorm, LayerNorm, GroupNorm (dense-path integrated) |
| `pkg/neuron` | Cell and axon primitives |
| `pkg/activation` | Activation functions, derivatives, vector softmax |
| `pkg/loss` | Loss functions and true derivatives (MSE, BCE, CCE, Huber, …) |
| `pkg/persistence` | JSON config + weight persistence (schema 1.1.0) |
| `pkg/checkpoint` | Resumable training snapshots with gzip cold tier |
| `pkg/dataset` | Slice, CSV, MNIST IDX, and prefetch dataset sources |
| `pkg/quantization` | Post-training int8 quantization (conv prefix + dense head) |
| `pkg/visualization` | Read-only HTTP observability endpoint (`/v1/*`) |
| `pkg/compute` | Pluggable backend interface + CPU reference backend |

## License

[MIT License](LICENSE).
