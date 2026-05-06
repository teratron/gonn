# Example: Sine Wave Regression

Fitting a continuous sine wave function using a neural network. This example demonstrates how to use the `TanH` activation function for smooth non-linear regression.

## Overview

The network learns to map a single input $x$ to its sine value $y = \sin(x)$ over the range $[0, 2\pi]$.

## Network Topology

- **Input**: 1 feature ($x$)
- **Hidden Layer 1**: 16 neurons with `TanH` activation
- **Hidden Layer 2**: 16 neurons with `TanH` activation
- **Output**: 1 neuron with `Linear` activation ($y$)

## Why TanH?

For smooth, periodic functions like sine, the Hyperbolic Tangent (`TanH`) often outperforms `ReLU` because its output and derivatives are continuous and smooth.

$$\text{TanH}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

## Features

- **Uniform Grid**: Generates 200 evenly-spaced points across the input domain.
- **MSE Loss**: Optimized using Mean Squared Error.
- **Hold-out Validation**: Trains on the first 80% of the grid and validates on the remaining 20% to test extrapolation/interpolation.

## Performance Target

The example aims for a test Root Mean Squared Error (RMSE) of $\leq 0.10$.

## Running the example

```bash
go run ./examples/regression_sin
```
