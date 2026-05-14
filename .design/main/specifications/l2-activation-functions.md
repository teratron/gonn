# Activation Functions

**Version:** 1.0.0
**Status:** Stable
**Layer:** implementation
**Implements:** l1-math-functions.md

## Overview

Specifies the implementation of activation functions for the GoNN neural network library. Provides 10 activation functions via a dispatcher pattern (switch on `Type` enum) and an optional `Function[T]` interface for object-oriented usage. Each function implements both forward (activation) and backward (derivative) computation.

## Related Specifications

- [l1-math-functions.md](l1-math-functions.md) — Parent concept spec
- [l2-loss-functions.md](l2-loss-functions.md) — Sibling: loss function implementations

## 1. Motivation

Activation functions introduce non-linearity into neural network computations. This package provides a comprehensive set of commonly used activation functions, all generic over `float32`/`float64`, with a consistent dispatcher API.

## 2. Constraints & Assumptions

- All functions are generic over `utils.Float`
- Dispatcher uses `Type uint8` enum with `String()` method
- Each function is in a separate file (e.g., `sigmoid.go`, `relu.go`)
- Derivative functions expect post-activation values as input
- `params ...float64` variadic for function-specific parameters (e.g., LeakyReLU leak factor)

## 4. Invariant Compliance

| L1 Invariant | Implementation |
| :--- | :--- |
| INV-1 (Activation + Derivative pair) | Each function file exports both `*Activation()` and `*Derivative()` |
| INV-2 (Dispatcher pattern) | `Activation(value, mode)` and `Derivative(value, mode)` switch on `Type` |
| INV-3 (Pure functions) | All implementations are stateless package-level functions |
| INV-4 (Dual API) | `Function[T]` interface exists alongside dispatcher (only Sigmoid implements it) |
| INV-5 (Post-activation derivative) | All derivative functions receive activated values |

## 5. Detailed Design

### 5.1 Type Enum

```plaintext
Type uint8
├── LINEAR    = 0
├── RELU      = 1
├── LeakyReLU = 2
├── SIGMOID   = 3
├── TanH      = 4
├── SWISH     = 5
├── ELU       = 6
├── SELU      = 7
├── ELISH     = 8
├── SOFTMAX   = 9
```

### 5.2 File Structure

```plaintext
pkg/activation/
├── activation.go       # Type enum, dispatcher, Function[T] interface
├── activation_test.go  # Tests (currently: only String() test)
├── sigmoid.go          # sigmoidActivation, sigmoidDerivative + Sigmoid[T] struct
├── tanh.go             # tanhActivation, tanhDerivative
├── relu.go             # reluActivation, reluDerivative (+ LeakyReLU via params)
├── elu.go              # eluActivation, eluDerivative
├── selu.go             # seluActivation, seluDerivative
├── swish.go            # swishActivation, swishDerivative
├── elish.go            # elishActivation, elishDerivative
├── linear.go           # linearActivation, linearDerivative
├── softmax.go          # softmaxActivation, softmaxDerivative (PLACEHOLDER)
└── README.md           # Package documentation
```

### 5.3 Function Interface

```plaintext
Function[T utils.Float]  (interface)
├── Activation(value *T)
└── Derivative(value *T)

Sigmoid[T utils.Float]  (struct, implements Function[T])
├── Activation(value *T)  — in-place: *value = 1/(1+exp(-*value))
└── Derivative(value *T)  — in-place: *value = *value * (1 - *value)
```

### 5.4 Known Issues

1. **Softmax is a placeholder** — implements `exp(x)/(exp(x)+1)` which is actually sigmoid. Real softmax requires vector input (entire layer values)
2. **SELU alpha typo** — constant `1.673263242354372848...` missing digit `7` → should be `1.6732632423543772848...`
3. **Only Sigmoid implements Function[T]** — 9 other functions only accessible via dispatcher
4. **Tests only check String()** — no mathematical correctness tests
5. **TanH derivative convention** — expects already-activated value (consistent with INV-5 but undocumented)
6. **LeakyReLU** — reuses `reluActivation()` with `leak` param; no separate struct

## 6. Implementation Notes

1. Implement real Softmax (requires layer-level vector computation, not scalar)
2. Fix SELU alpha constant
3. Add table-driven tests for each function with known input/output pairs
4. Consider removing Function[T] interface or implementing it for all functions
5. Add documentation for derivative input convention

## 7. Drawbacks & Alternatives

- **Drawback**: Dual API (interface + dispatcher) adds confusion — only one is actually used
- **Alternative**: Remove Function[T] interface, keep only dispatcher — simpler
- **Alternative**: Strategy pattern (one struct per function) — more extensible but more heap allocations

## Canonical References

| Alias | Path | Purpose |
| :--- | :--- | :--- |
| `[ACT]` | `pkg/activation/activation.go` | Dispatcher, Type enum, Function interface |
| `[SIGMOID]` | `pkg/activation/sigmoid.go` | Reference implementation (interface + dispatcher) |
| `[ACT_TEST]` | `pkg/activation/activation_test.go` | Existing tests (String only) |
| `[README]` | `pkg/activation/README.md` | Package documentation |

## Document History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0.0 | 2026-04-21 | Initial Stable — reverse-engineered from existing codebase |
