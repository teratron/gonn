# Activation Functions

Package `activation` provides a suite of activation functions and their derivatives for neural network computations. It implements the **Dispatcher Pattern (C27)**, allowing for efficient, enum-based function selection with zero external dependencies.

> [!NOTE]
> All functions are generic and support both `float32` and `float64` via the `utils.Float` constraint (C25).

## Core API

The package exposes two primary entry points that dispatch to specific implementations based on the `activation.Type` enum.

```go
import "github.com/teratron/gonn/pkg/activation"

// 1. Calculate activation
y := activation.Activation(x, activation.ReLU)

// 2. Calculate derivative (used during backpropagation)
dy := activation.Derivative(x, activation.ReLU)
```

## Available Functions

| Identifier | Description | Formula | Parameters |
| :--- | :--- | :--- | :--- |
| `ELISH` | Exponential Linear Unit + Sigmoid | $f(x) = \begin{cases} x \sigma(x) & x \ge 0 \\ (e^x - 1) \sigma(x) & x < 0 \end{cases}$ | - |
| `ELU` | Exponential Linear Unit | $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$ | `alpha` (default: `1.0`) |
| `Linear` | Identity function | $f(x) = \text{slope} \cdot x + \text{offset}$ | `slope` (1.0), `offset` (0.0) |
| `LeakyReLU` | Leaky Rectified Linear Unit | $f(x) = \begin{cases} x & x > 0 \\ \text{leak} \cdot x & x \le 0 \end{cases}$ | `leak` (default: `0.01`) |
| `ReLU` | Rectified Linear Unit | $f(x) = \max(0, x)$ | - |
| `SELU` | Scaled Exponential Linear Unit | $f(x) = \lambda \cdot \text{ELU}(x, \alpha)$ | $\lambda \approx 1.0507$, $\alpha \approx 1.6733$ |
| `SIGMOID` | Logistic / Sigmoid | $f(x) = \frac{1}{1 + e^{-\text{slope} \cdot x}}$ | `slope` (default: `1.0`) |
| `SOFTMAX` | Softmax activation | $f(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ | - |
| `SWISH` | Swish-function | $f(x) = x \cdot \sigma(\beta x)$ | `beta` (default: `1.0`) |
| `TanH` | Hyperbolic Tangent | $f(x) = \tanh(x)$ | - |

> [!WARNING]
> **Softmax Implementation Note**: The current implementation of `SOFTMAX` in this package acts as a placeholder for single-value operations. Full vector-based Softmax (which requires normalized outputs across a layer) is typically handled at the layer or network level.

## Implementation Details

Following project convention **C27**, each activation function is implemented in its own file (e.g., `relu.go`, `sigmoid.go`) and registered in the central dispatcher within `activation.go`.

- **Generic Support**: All functions accept `T utils.Float`.
- **Zero Allocations**: Functions are designed for high-performance inner loops.
- **Derivatives**: Every function has a corresponding derivative implementation for gradient descent.
