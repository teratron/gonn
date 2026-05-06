# Example: Multi-output Regression

Learning a complex 5-dimensional input to 3-dimensional output mapping using synthetic data. This example demonstrates `ReLU` activation combined with `He` weight initialization for regression tasks.

## Overview

The network is tasked with learning three distinct mathematical functions simultaneously from a common pool of 5 input variables $x \in [-1, 1]^5$.

## Target Functions

The network learns to approximate the following ground-truth mapping:

1. **Linear**: $y_0 = x_0 + x_1$
2. **Bilinear**: $y_1 = x_2 \cdot x_3$
3. **Quadratic**: $y_2 = 0.5 \cdot (x_4^2 - x_0)$

## Network Topology

- **Input**: 5 features
- **Hidden Layer 1**: 16 neurons with `ReLU` activation
- **Hidden Layer 2**: 8 neurons with `ReLU` activation
- **Output**: 3 neurons with `Linear` activation

## Features

- **He Initialization**: Uses `WeightInitHe`, which is optimized for `ReLU` layers to prevent vanishing/exploding gradients in deep networks.
- **RMSE Monitoring**: Reports Root Mean Squared Error (RMSE) independently for each output dimension.
- **Synthetic Data**: Generates 500 samples on-the-fly with a deterministic seed for reproducibility.

## Mathematical Note: RMSE

The performance is measured using RMSE for each dimension $d$:
$$RMSE_d = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{i,d} - \hat{y}_{i,d})^2}$$

Target performance is a test RMSE of $\leq 0.20$ across all dimensions.

## Running the example

```bash
go run ./examples/regression_multi
```
