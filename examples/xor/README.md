# Example: XOR

The classic XOR problem — the smallest non-linear classification task. This example proves that backpropagation correctly wires the hidden layer to solve the problem.

## Network Topology

The architecture follows a minimal 1-hidden-layer setup:

- **Input**: 2 neurons (X, Y binary inputs)
- **Hidden Layer**: 4 neurons with `Sigmoid` activation
- **Output**: 1 neuron with `Sigmoid` activation

## Features

- **API Parity**: Compares **Builder API** and **Functional Options API** side-by-side using the same hyperparameters.
- **Dataset**: Uses the standard XOR truth table (4 samples).
- **Training**: Demonstrates the `Fit` method for batch training until convergence.
- **Initialization**: Showcases `Xavier` weight initialization.

## Running the example

```bash
go run ./examples/xor
```

## Running the tests

```bash
go test ./examples/xor
```
