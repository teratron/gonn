# Example: Precision Parity

This example compares the performance and accuracy of `float32` vs `float64` implementations. It builds and trains the same XOR network topology using both numeric precisions.

## Network Topology

Both networks share an identical 1-hidden-layer structure:

- **Input**: 2 neurons
- **Hidden Layer**: 4 neurons with `Sigmoid` activation
- **Output**: 1 neuron with `Sigmoid` activation

## Features

- **Generics Support**: Demonstrates how `GoNN` uses Go generics to support different numeric types (`float32` and `float64`) without code duplication in the core library.
- **Side-by-side Comparison**: Measures and compares training loss and execution time for both precisions.
- **Parity Baseline**: Validates that both implementations converge to similar loss values under identical hyperparameters.

## Running the example

```bash
go run ./examples/precision
```

## Running the tests

```bash
go test ./examples/precision
```
