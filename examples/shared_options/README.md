# Example: Shared Options

This example demonstrates how to share a common set of `nn.Option[T]` across multiple network instances with different topologies.

## Overview

In complex projects, many hyperparameters (like learning rate, loss function, and iteration limits) are often shared across different network architectures. `GoNN` allows you to define these options once and reuse them.

## Topologies

This example trains two different XOR-capable architectures using shared hyperparameters:

1. **Topology A**: 2 Input → 8 ReLU (Hidden) → 1 Sigmoid (Output)
2. **Topology B**: 2 Input → 16 ReLU (Hidden) → 1 Sigmoid (Output)

## Features

- **Option Composition**: Demonstrates defining a shared `[]Option[T]` slice and appending architecture-specific options.
- **Hyperparameter Consistency**: Ensures that both Topology A and Topology B are trained with identical learning rates and convergence criteria.
- **Side-by-side Comparison**: Compares training performance (epochs and loss) between the two topologies.

## Running the example

```bash
go run ./examples/shared_options
```

## Running the tests

```bash
go test ./examples/shared_options
```
