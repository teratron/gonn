# GoNN Examples Catalog

Implements the v0.1 subset of [`l2-usage-examples.md`](../.design/specifications/l2-usage-examples.md). Every active example is its own Go module with a smoke test gated on `go test ./examples/...`.

## v0.1 Active

| ID | Path | Style | Demonstrates |
| :--- | :--- | :--- | :--- |
| E01 | [xor/](xor/) | Builder + Options | Canonical XOR — dual-style parity baseline |
| E02 | [logic_gates/](logic_gates/) | Builder | AND / OR / NAND truth-table fits |
| E09 | [persistence/](persistence/) | Builder + `pkg/persistence` | Save → reload → query, ULP-1 round-trip (PERS-4) |
| E11 | [callbacks/](callbacks/) | Options | `WithEpochCallback` + `WithBatchCallback` |
| E12 | [style_showcase/](style_showcase/) | Builder + Options + Preset | Three-style equivalence on the same network |
| E14 | [shared_options/](shared_options/) | Options | Shared `[]Option[T]` across two single-hidden topologies (adapted from spec Topology B) |
| E15 | [precision/](precision/) | Builder | `float32` vs `float64` parity at identical hyperparameters |

Run any one example: `go run ./examples/{name}/`. Run the whole smoke suite: `go test ./examples/...`.

## Deferred to v0.2

These catalog entries stay in the spec but require features not yet in v0.1. Each unblocks once its gate clears.

| ID | Path | Gate |
| :--- | :--- | :--- |
| E03 | [perceptron/](perceptron/) | v0.2 multi-hidden (file currently runs a single-hidden stub; `// removed once v0.2 lands`) |
| E04 | `examples/binary_classification/` | v0.2 multi-hidden |
| E05 | `examples/iris/` | v0.2 multi-hidden |
| E06 | `examples/mnist/` | v0.2 multi-hidden + dataset-loader spec |
| E07 | `examples/regression_sin/` | v0.2 multi-hidden |
| E08 | `examples/regression_multi/` | v0.2 multi-hidden + `PresetRegression` extension |
| E10 | `examples/continuation/` | `AndTrain` API surface |
| E13 | `examples/higher_order_options/` | v0.2 multi-hidden |

## Coverage Matrix Audit

Cross-reference of `l2-usage-examples` §5.3 against the v0.1 active examples. **`covered`** means at least one active example exercises the API element; **`v0.2`** means the remaining covering examples are deferred.

| API Element | v0.1 status | Active examples | Notes |
| :--- | :--- | :--- | :--- |
| `NewBuilder[T]()` + `Compile()` | covered | E01, E02, E12, E15 | E03/E07/E10 deferred |
| `New[T](opts...)` / `MustNew[T]` | covered | E01, E11, E12, E14 | E04/E05/E08/E13 deferred |
| `Input` / `WithInput` | covered | all v0.1 actives | — |
| `Dense` / `WithHiddenLayer` | covered | all v0.1 actives | — |
| `Output` / `WithOutput` | covered | all v0.1 actives | — |
| `WithLearningRate` | covered | all v0.1 actives | — |
| `WithLoss` | covered | all v0.1 actives | — |
| `WithBias` | covered | E11 (preset → bias on), E14 | E03 deferred |
| `WithWeightInit` (xavier/he) | covered | E01 (xavier) | E04 he, E06 he, E07 xavier deferred |
| `WithMaxIterations` | covered | all v0.1 actives | — |
| `WithLossLimit` | covered | E01, E12, E14 | — |
| `WithEpochCallback` | covered | E11 | E05 deferred |
| `WithBatchCallback` | covered | E11 | — |
| `Sequential` | **gap → v0.2** | (none) | E13 deferred |
| `DeepNetwork` | **gap → v0.2** | (none) | E13 deferred |
| `PresetXOR` | covered | E11 (preset), E12 | — |
| `PresetMNIST` | **gap → v0.2** | (none) | E06 deferred (multi-hidden + loader) |
| `PresetRegression` | **gap → v0.2** | (none) | E08 deferred (multi-hidden) |
| `Train` | covered | all v0.1 actives via `Fit` | — |
| `Query` | covered | E01, E09, E11, E12, E14, E15 | — |
| `Verify` | **gap → v0.2** | (none) | E04 / E05 / E07 deferred |
| `AndTrain` | **gap → v0.2** | (none) | API surface not yet defined |
| `Persistence` (Save / Reload) | covered (manual seam) | E09 | future `nn.Save` / `nn.Load` will close the seam |

**v0.1 gap summary**: 6 elements (`Sequential`, `DeepNetwork`, `PresetMNIST`, `PresetRegression`, `Verify`, `AndTrain`) are uncovered until the multi-hidden / `AndTrain` / dataset-loader work lands. None are blocking the v0.1 release bar.
