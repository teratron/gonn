# Loss Functions

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-math-functions.md

## Overview

Specifies the implementation of loss (error) functions for the GoNN neural network library. Provides 18 loss functions via a dispatcher pattern and a `CalculateTotalLoss` aggregator. Each function computes the error between a predicted value and a target value.

## Related Specifications

- [l1-math-functions.md](l1-math-functions.md) — Parent concept spec
- [l2-activation-functions.md](l2-activation-functions.md) — Sibling: activation function implementations

## 1. Motivation

Loss functions quantify how far the network's predictions are from the target values. The package provides a wide selection of loss functions for different use cases (regression, classification, robust estimation), all accessible via a consistent dispatcher API.

## 2. Constraints & Assumptions

- All functions are generic over `utils.Float`
- Dispatcher uses `Type uint8` enum with `String()` method
- Each function is in a separate file
- Functions operate on scalar pairs (predicted, target) — no batch/vector mode
- `CalculateTotalLoss` aggregates individual cell losses

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-2 (Dispatcher pattern) | `Loss(predicted, target, mode)` switches on `Type` |
| INV-3 (Pure functions) | All implementations are stateless package-level functions |
| INV-1 (Generic Float) | All functions parameterized by `T utils.Float` |

## 5. Detailed Design

### 5.1 Type Enum (18 functions)

```plaintext
Type uint8
├── MSE         = 0   Mean Squared Error
├── MAE         = 1   Mean Absolute Error
├── RMSE        = 2   Root Mean Squared Error
├── ARCTAN      = 3   Arctangent Loss
├── AVG         = 4   Average (same as MAE)
├── BCE         = 5   Binary Cross-Entropy
├── CCE         = 6   Categorical Cross-Entropy
├── COSINE      = 7   Cosine Similarity (MISSING implementation)
├── HINGE       = 8   Hinge Loss
├── SQ_HINGE    = 9   Squared Hinge Loss
├── CAT_HINGE   = 10  Categorical Hinge Loss
├── HUBER       = 11  Huber Loss
├── KLD         = 12  KL Divergence
├── LOG_COSH    = 13  Log-Cosh Loss
├── MAPE        = 14  Mean Absolute Percentage Error
├── MSLE        = 15  Mean Squared Logarithmic Error
├── POISSON     = 16  Poisson Loss
```

### 5.2 File Structure

```plaintext
pkg/loss/
├── loss.go          # Type enum, Loss() dispatcher, CalculateTotalLoss()
├── loss_test.go     # Tests (currently: only String() test)
├── mse.go           # mseLoss
├── mae.go           # maeLoss
├── rmse.go          # rmseLoss
├── arctan.go        # arctanLoss
├── avg.go           # avgLoss (duplicate of MAE)
├── bce.go           # bceLoss, bceLossSingle
├── cce.go           # cceLoss, cceLossSingle (returns 0)
├── hinge.go         # hingeLoss
├── sq_hinge.go      # sqHingeLoss
├── cat_hinge.go     # catHingeLoss
├── huber.go         # huberLoss
├── kld.go           # kldLoss
├── log_cosh.go      # logCoshLoss
├── mape.go          # mapeLoss
├── msle.go          # msleLoss
├── poisson.go       # poissonLoss
└── README.md        # Package documentation
```

### 5.3 Aggregator

```plaintext
CalculateTotalLoss[T](misses *[]*T, mode Type) T
  For each miss in misses:
    total += Loss(0.0, *miss, mode)
  return total / len
```

### 5.4 Known Issues

1. **COSINE enum exists but no implementation** — falls through to default (MSE) in switch
2. **MAE and AVG are identical** — `|predicted - target|`, pure duplication
3. **`cceLossSingle()` always returns 0** — scalar CCE is meaningless (needs vector)
4. **`CalculateTotalLoss` semantic error** — passes `0.0` as predicted and `miss` as target, but miss = target - predicted, not target
5. **`LossVector` commented out** — no batch operation support
6. **No cosine.go file** exists despite COSINE constant in enum
7. **Tests only check String()** — no mathematical correctness tests

## 6. Implementation Notes

1. Implement `cosine.go` or remove COSINE from enum
2. Remove or alias AVG to MAE to eliminate duplication
3. Fix `CalculateTotalLoss` semantics (predicted vs miss confusion)
4. Implement `cceLossSingle` properly or document that CCE requires vector mode
5. Add table-driven tests with known mathematical values

## 7. Drawbacks & Alternatives

- **Drawback**: 18 functions with only scalar operations limits usefulness for classification tasks
- **Alternative**: Add vector-based loss computation for CCE, Softmax+CCE combo
- **Drawback**: Some functions (COSINE, CCE scalar) are non-functional stubs

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[LOSS]` | `pkg/loss/loss.go` | Dispatcher, Type enum, CalculateTotalLoss |
| `[LOSS_TEST]` | `pkg/loss/loss_test.go` | Existing tests (String only) |
| `[MSE]` | `pkg/loss/mse.go` | Reference loss implementation |
| `[README]` | `pkg/loss/README.md` | Package documentation |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-21 | Initial Stable — reverse-engineered from existing codebase |
