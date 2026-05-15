# Example: Continuation

Demonstrates the `AndTrain` method for continuation training. The example
trains a 2→4→1 XOR network on the canonical truth table, then calls
`AndTrain` with a negated dataset and a smaller learning rate. The network's
existing weights are reused — `AndTrain` only resets convergence counters,
not weights — so the post-`AndTrain` predictions shift toward the negated
relationship without rebuilding the network or losing learned structure.

## Spec coverage

| Invariant | Where demonstrated |
| :--- | :--- |
| FMT-6 — AndTrain resets convergence counters but preserves weights | Phase 2 in `main.go`: same `*NN[T]` instance, no weight reinit |
| FMT-7 — Callbacks remain active across AndTrain | If `WithOnIterationEnd` is added to the base network, it stays live during `AndTrain` |
| FMT-8 — AndTrain refuses while training is running | Returns `utils.ErrNetworkRunning` (covered by `pkg/nn/andtrain_test.go::TestAndTrainRejectsRunning`) |
| Base config restored on return | `nn.cfg.LearningRate` is back to 0.3 after the `AndTrain(...WithLearningRate(0.05))` call |

## Run

```bash
go run ./examples/continuation/
```

Expected output (weight init is seeded by wall-clock; numbers vary):

```
Phase 1 (XOR): 2000 epochs, loss=0.001
Phase 1 predictions: [0.041 0.953 0.948 0.052]
Phase 2 (negated, AndTrain): 2000 epochs, loss=0.001
Phase 2 predictions: [0.949 0.046 0.052 0.957]
```

The `Phase 2 predictions` row is the inverse of `Phase 1 predictions` — the
network re-learned the negated relationship using the same weight slab.

## Test

```bash
go test ./examples/continuation/
```

Asserts that the post-`AndTrain` predictions are closer to the negated
targets than to the original XOR targets.
