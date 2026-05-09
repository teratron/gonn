# Example: Binary Classification

This example demonstrates how to build and train a multi-hidden layer neural network for binary classification on a synthetic Gaussian-blob dataset using functional options.

## Features

- **Options API**: Network construction using functional options (`nn.New`).
- **Multi-Hidden Topology**: Demonstrates a deeper network `2 → ReLU(8) → ReLU(8) → Sigmoid(1)`.
- **Custom Dataset**: Generates a reproducible 2-D Gaussian-blob dataset (200 points, 2 classes) using PCG PRNG.
- **Advanced Configuration**: Showcases `WithLoss` (`BCE`), `WithWeightInit` (`nn.WeightInitHe`), and customized `WithLearningRate`.
- **Metrics Evaluation**: Uses a custom helper function to evaluate classification accuracy over a hold-out test set (80/20 train/test split).

## Running the example

```bash
go run ./examples/binary_classification
```

## Running the tests

```bash
go test ./examples/binary_classification
```
