# GoNN Examples Catalog

Implements the v0.2 multi-hidden subset of [`l2-usage-examples.md`](../.design/specifications/l2-usage-examples.md). Every active example is its own Go module with a smoke test.

## Active

| ID | Path | Style | Demonstrates |
| :--- | :--- | :--- | :--- |
| E01 | [xor](xor/) | Builder + Options | Canonical XOR — dual-style parity baseline |
| E02 | [logic_gates](logic_gates/) | Builder | AND / OR / NAND truth-table fits |
| E03 | [perceptron](perceptron/) | Builder | 4-hidden mixed-activation topology (3→Sigmoid(5)→ReLU(10)→Sigmoid(5)→SoftMax(2)) |
| E04 | [binary_classification](binary_classification/) | Options | Multi-hidden BCE + He init; Gaussian-blob 2-class dataset |
| E05 | [iris](iris/) | Options | Multi-hidden MSE + EpochCallback; Fisher iris 3-class, `go:embed` CSV |
| E07 | [regression_sin](regression_sin/) | Options | 2-hidden TanH sine regression; RMSE target ≤ 0.10 |
| E08 | [regression_multi](regression_multi/) | Options | 2-hidden ReLU+He; 5-input → 3-output regression, per-dim RMSE ≤ 0.20 |
| E09 | [persistence](persistence/) | Builder + `pkg/persistence` | Save → reload → query, ULP-1 round-trip (PERS-4) |
| E11 | [callbacks](callbacks/) | Options | `WithEpochCallback` + `WithBatchCallback` |
| E12 | [style_showcase](style_showcase/) | Builder + Options + Preset | Three-style equivalence on the same network |
| E13 | [higher_order_options](higher_order_options/) | Options | `Sequential` + `DeepNetwork` higher-order options; iris CSV reuse |
| E14 | [shared_options](shared_options/) | Options | Shared `[]Option[T]` across two topologies |
| E15 | [precision](precision/) | Builder | `float32` vs `float64` parity at identical hyperparameters |

Run any one example: `go run ./examples/{name}/`. Run the whole smoke suite from the workspace root: `go test ./...`.

## Deferred

| ID | Path | Gate |
| :--- | :--- | :--- |
| E06 | `examples/mnist/` | dataset-loader spec + MNIST download |
| E10 | `examples/continuation/` | `AndTrain` API surface |

## Coverage Matrix

Cross-reference of `l2-usage-examples` §5.3 against active examples. **`covered`** means at least one active example exercises the element.

| API Element | Status | Active examples | Notes |
| :--- | :--- | :--- | :--- |
| `NewBuilder[T]()` + `Compile()` | covered | E01, E02, E03, E12, E15 | — |
| `New[T](opts...)` / `MustNew[T]` | covered | E01, E04, E05, E07, E08, E11, E12, E13, E14 | — |
| `Input` / `WithInput` | covered | all actives | — |
| `Dense` / `WithHiddenLayer` (multi) | covered | E03, E04, E05, E07, E08 | — |
| `Output` / `WithOutput` | covered | all actives | — |
| `WithLearningRate` | covered | all actives | — |
| `WithLoss` (MSE/BCE/ARCTAN) | covered | E04 (BCE), E05 (MSE), E03 (ARCTAN) | — |
| `WithBias` | covered | E04, E05, E07, E08, E11, E13, E14 | — |
| `WithWeightInit` (xavier/he) | covered | E01 (xavier), E04 (he), E07 (xavier), E08 (he) | — |
| `WithMaxIterations` | covered | all actives | — |
| `WithLossLimit` | covered | E01, E03, E12, E14 | — |
| `WithEpochCallback` | covered | E05, E11 | — |
| `WithBatchCallback` | covered | E11 | — |
| `Sequential` | covered | E13 | — |
| `DeepNetwork` | covered | E13 | — |
| `PresetXOR` | covered | E11, E12 | — |
| `PresetMNIST` | gap | (none) | E06 deferred (loader) |
| `PresetRegression` | gap | (none) | requires E08 extension |
| `Train` | covered | all actives via `Fit` | — |
| `Query` | covered | E01, E04, E05, E07, E08, E09, E11, E12, E13, E14, E15 | — |
| `Verify` | gap | (none) | planned E05/E07 extension |
| `AndTrain` | gap | (none) | E10 deferred |
| `Persistence` (Save / Reload) | covered | E09 | — |

**Gap summary**: 4 elements (`PresetMNIST`, `PresetRegression`, `Verify`, `AndTrain`) remain uncovered pending E06/E10/loader work.
