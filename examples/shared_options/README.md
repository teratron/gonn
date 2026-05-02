# Example: Shared Options

This example demonstrates how to share a common set of `nn.Option[T]` across multiple network instances with different topologies.

## Features

- Definition of a shared option slice.
- Topologies A and B inherit common hyperparameters (learning rate, loss, iterations).
- Side-by-side training comparison.

## Running the example

```bash
go run main.go
```
