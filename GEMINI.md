# GoNN (Go Neural Network) Development Guidelines

This document provides instructions for developers working on the GoNN library.

## Project Overview

GoNN (Go Neural Network) is a library written in Go for creating and training neural networks. The project is in its early stages and aims to provide the fundamental components for building various neural network architectures.

The core logic is located in the `pkg` directory, which is organized into modules for different functionalities:

- `activation`: Contains activation functions (e.g., Sigmoid, ReLU, Softmax).
- `loss`: Contains loss functions (e.g., MSE, Cross-Entropy).
- `layer`: Defines the different types of layers (e.g., Dense, Input, Output).
- `network`: Handles the neural network structure, training, and propagation.
- `nn`: Provides a high-level fluent API (`builder.go`) for constructing and configuring neural networks.

## Building and Running

### Build

To build the entire library, use the standard Go build command from the root directory:

```shell
go build ./...
```

### Test

To run all tests for the library, use the following command:

```shell
go test ./...
```

### Run Examples

The project includes examples in the `examples` directory. To run a specific example, navigate to its directory and use `go run`. For instance, to run the perceptron example:

```shell
cd examples/perceptron
go run main.go
```

## Development Conventions

### Fluent API

The library uses a fluent API for building neural network models, which allows for chaining method calls to define the network's architecture.

**Example:**

```go
import "github.com/teratron/gonn/pkg/nn"
import "github.com/teratron/gonn/pkg/activation"
import "github.com/teratron/gonn/pkg/loss"

n := nn.New[float32]().
    Input(3).
    Dense(5, activation.SIGMOID, true).
    Dense(10, activation.ReLU, true).
    Output(2, activation.SOFTMAX, loss.ARCTAN, true)
```

### Code Style

Follow the standard Go conventions (`gofmt`). The codebase is structured in a modular way within the `pkg` directory. When adding new features, adhere to the existing modular structure. For example, new activation functions should be added to the `pkg/activation` directory.
