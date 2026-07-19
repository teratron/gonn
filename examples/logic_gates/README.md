# Example: Logic Gates Suite

This example shows how a small neural network can specialize to learn different logical functions (AND, OR, NAND) based on the provided truth tables.

## Network Topology

The network uses a minimal architecture capable of learning linearly separable logical functions:

- **Input**: 2 neurons
- **Hidden Layer**: 2 neurons with `Sigmoid` activation
- **Output**: 1 neuron with `Sigmoid` activation

## Features

- **Multi-task Learning**: Trains the same topology on different datasets (AND, OR, NAND) in sequence.
- **Parameter Specialization**: Illustrates how weights specialize to different logical functions.
- **Linear Separability**: Demonstrates simple classification tasks (XOR is excluded as it requires more neurons).

## Running the example

```bash
go run ./examples/logic_gates
```

## Running the tests

```bash
go test ./examples/logic_gates
```
