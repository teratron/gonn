# Example: Perceptron

A multi-layer perceptron demonstrating a deep topology using the Builder API. This example solves a synthetic regression task using a sliding window over a numeric stream.

## Network Topology

The architecture follows a 4-hidden-layer stack with mixed activations and bias configurations:

- **Input**: 3 neurons
- **Hidden Layer 1**: 5 neurons with `Sigmoid` activation (with bias)
- **Hidden Layer 2**: 10 neurons with `ReLU` activation (with bias)
- **Hidden Layer 3**: 5 neurons with `Sigmoid` activation (no bias)
- **Output**: 2 neurons with `SoftMax` activation (with bias)

## Features

- **Builder API**: Uses the fluent `Builder` interface for clear, step-by-step network assembly.
- **Mixed Bias**: Demonstrates how to enable or disable bias per-layer.
- **Custom Loss**: Uses the `ARCTAN` loss function.
- **Xavier Initialization**: Uses Xavier weight initialization for stability in deep networks.

## Running the example

```bash
go run ./examples/perceptron
```

## Running the tests

```bash
go test ./examples/perceptron
```
