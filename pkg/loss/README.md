# Loss Functions

Package `loss` provides a variety of cost functions used to quantify the difference between predicted values and actual targets. It implements the **Dispatcher Pattern (C27)** for standardized function selection and high-performance computation.

> [!NOTE]
> All loss functions are generic and support both `float32` and `float64` via the `utils.Float` constraint (C25).

## Core API

The package dispatches calculations through the central `Loss` function for individual values, or `CalculateTotalLoss` for accumulated errors over a batch.

```go
import "github.com/teratron/gonn/pkg/loss"

// 1. Calculate loss for a single prediction
l := loss.Loss(predicted, target, loss.MSE)

// 2. Calculate average loss over a set of misses (errors)
total := loss.CalculateTotalLoss(misses, loss.MSE)
```

## Available Functions

| Identifier | Description | Formula | Implementation |
| :--- | :--- | :--- | :--- |
| `MSE` | Mean Squared Error | $L = (y_{pred} - y_{true})^2$ | Scalar |
| `MAE` / `AVG` | Mean Absolute Error | $L = \|y_{pred} - y_{true}\|$ | Scalar |
| `BCE` | Binary Cross-Entropy | $L = -(y_{true} \ln(y_{pred}) + (1-y_{true}) \ln(1-y_{pred}))$ | Scalar |
| `RMSE` | Root Mean Squared Error | $L = \sqrt{(y_{pred} - y_{true})^2}$ | Scalar |
| `MSLE` | Mean Squared Logarithmic Error | $L = (\ln(y_{pred} + 1) - \ln(y_{true} + 1))^2$ | Scalar |
| `MAPE` | Mean Absolute Percentage Error | $L = \left\| \frac{y_{true} - y_{pred}}{y_{true}} \right\| \cdot 100$ | Scalar |
| `KLD` | Kullback-Leibler Divergence | $L = y_{true} \ln\left(\frac{y_{true}}{y_{pred}}\right)$ | Scalar |
| `POISSON` | Poisson Loss | $L = y_{pred} - y_{true} \ln(y_{pred})$ | Scalar |
| `HINGE` | Hinge Loss | $L = \max(0, 1 - y_{pred} \cdot y_{true})$ | Scalar |
| `SQ_HINGE` | Squared Hinge Loss | $L = \max(0, 1 - y_{pred} \cdot y_{true})^2$ | Scalar |
| `LOG_COSH` | Log-Cosh Loss | $L = \ln(\cosh(y_{pred} - y_{true}))$ | Scalar |
| `HUBER` | Huber Loss | $L = \begin{cases} 0.5(y_p - y_t)^2 & \|y_p - y_t\| \le \delta \\ \delta(\|y_p - y_t\| - 0.5\delta) & \text{otherwise} \end{cases}$ | Scalar |
| `ARCTAN` | Arctan Error | $L = \arctan(y_{pred} - y_{true})$ | Scalar |
| `CCE` | Categorical Cross-Entropy | $L = -\sum y_{true} \ln(y_{pred})$ | Vector |
| `COSINE` | Cosine Distance | $L = 1 - \frac{y_{pred} \cdot y_{true}}{\|y_{pred}\| \|y_{true}\|}$ | Vector |

> [!IMPORTANT]
> **Vector Operations**: While most functions are implemented for scalar inputs, `CCE` and `COSINE` are inherently vector-based. In the current scalar `Loss` dispatcher, they may return placeholder values (0.0) or require explicit slice handling via `LossVector` (planned).

## Implementation Details

Following project convention **C27**, each loss function is isolated in its own file (e.g., `mse.go`, `huber.go`) and registered in `loss.go`.

- **Numerical Stability**: Logarithmic and division-based functions (BCE, KLD, MAPE) include an $\epsilon \approx 1e-7$ clamp to prevent `NaN` or `Inf` results.
- **Generic Architecture**: Leverages Go generics to ensure precision consistency across the library.
- **Total Loss**: `CalculateTotalLoss` handles averaging and final transformations (e.g., square roots for RMSE) over a batch of error values.
