# Example: Higher-order Options

Demonstrates the power of the Functional Options API by using higher-order builders like `Sequential` and `DeepNetwork` to compose complex topologies concisely.

## Overview

In `GoNN`, options are not just configuration flags; they are functions that can be composed. This example showcases:

1. `Sequential`: Creating multiple identical hidden layers.
2. `DeepNetwork`: Creating a deep architecture where each layer's size is a fraction of the previous one.

## Topologies

### 1. Sequential Variant

Defined as `Sequential(2, 16, SIGMOID)`, which expands to:

- Hidden Layer 1: 16 neurons, Sigmoid
- Hidden Layer 2: 16 neurons, Sigmoid

### 2. DeepNetwork Variant

Defined as `DeepNetwork(32, 3, SIGMOID)`, which implements a "pyramid" decay:

- Layer 1: 32 neurons
- Layer 2: 16 neurons (32/2)
- Layer 3: 8 neurons (16/2)

## Features

- **Feature Normalization**: Uses Min-Max scaling to map input features to the $[0, 1]$ range, improving training stability for deeper networks.
- **API Composition**: Shows how these higher-order helpers coexist with standard options like `WithBias` and `WithLearningRate`.
- **Iris Reuse**: Validates these topologies against the Fisher Iris dataset.

## Mathematical Note: Min-Max Scaling

The inputs are normalized using the following formula:
$$x' = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}$$

This ensures that all input features have the same scale, preventing features with large magnitudes from dominating the gradient updates.

## Running the example

```bash
go run ./examples/higher_order_options
```
