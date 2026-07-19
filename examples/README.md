# GoNN Examples Catalog

Implements various usage patterns for the GoNN library. Every active example is its own Go module with a smoke test.

## Active

| Path | Style | Demonstrates |
| :--- | :--- | :--- |
| [xor](xor/) | Builder + Options | Canonical XOR — dual-style parity baseline |
| [logic_gates](logic_gates/) | Builder | AND / OR / NAND truth-table fits |
| [perceptron](perceptron/) | Builder | 4-hidden mixed-activation topology (3→Sigmoid(5)→ReLU(10)→Sigmoid(5)→SoftMax(2)) |
| [binary_classification](binary_classification/) | Options | Multi-hidden BCE + He init; Gaussian-blob 2-class dataset |
| [iris](iris/) | Options | Multi-hidden MSE + EpochCallback; Fisher iris 3-class, `go:embed` CSV |
| [regression_sin](regression_sin/) | Options | 2-hidden TanH sine regression; RMSE target ≤ 0.10 |
| [regression_multi](regression_multi/) | Options | 2-hidden ReLU+He; 5-input → 3-output regression, per-dim RMSE ≤ 0.20 |
| [persistence](persistence/) | Builder + `pkg/persistence` | Save → reload → query, ULP-1 round-trip |
| [continuation](continuation/) | Options | `AndTrain` continuation — weight reuse parity |
| [callbacks](callbacks/) | Options | `WithEpochCallback` + `WithBatchCallback` |
| [style_showcase](style_showcase/) | Builder + Options + Preset | Three-style equivalence on the same network |
| [higher_order_options](higher_order_options/) | Options | `Sequential` + `DeepNetwork` higher-order options; iris CSV reuse |
| [shared_options](shared_options/) | Options | Shared `[]Option[T]` across two topologies |
| [precision](precision/) | Builder | `float32` vs `float64` parity at identical hyperparameters |

Run any one example: `go run ./examples/{name}/`. Run the whole smoke suite from the workspace root: `go test ./...`.

## Deferred

| Path | Gate |
| :--- | :--- |
| `examples/mnist/` | dataset-loader spec + MNIST download |

## Coverage Matrix

Cross-reference of library features against active examples. **`covered`** means at least one active example exercises the element.

| API Element | Status | Active examples | Notes |
| :--- | :--- | :--- | :--- |
| `NewBuilder[T]()` + `Compile()` | covered | xor, logic_gates, perceptron, style_showcase, precision | — |
| `New[T](opts...)` / `MustNew[T]` | covered | xor, binary_classification, iris, regression_sin, regression_multi, callbacks, style_showcase, higher_order_options, shared_options | — |
| `Input` / `WithInput` | covered | all actives | — |
| `Dense` / `WithHiddenLayer` (multi) | covered | perceptron, binary_classification, iris, regression_sin, regression_multi | — |
| `Output` / `WithOutput` | covered | all actives | — |
| `WithLearningRate` | covered | all actives | — |
| `WithLoss` (MSE/BCE/ARCTAN) | covered | binary_classification (BCE), iris (MSE), perceptron (ARCTAN) | — |
| `WithBias` | covered | binary_classification, iris, regression_sin, regression_multi, callbacks, higher_order_options, shared_options | — |
| `WithWeightInit` (xavier/he) | covered | xor (xavier), binary_classification (he), regression_sin (xavier), regression_multi (he) | — |
| `WithMaxIterations` | covered | all actives | — |
| `WithLossLimit` | covered | xor, perceptron, style_showcase, shared_options | — |
| `WithEpochCallback` | covered | iris, callbacks | — |
| `WithBatchCallback` | covered | callbacks | — |
| `Sequential` | covered | higher_order_options | — |
| `DeepNetwork` | covered | higher_order_options | — |
| `PresetXOR` | covered | callbacks, style_showcase | — |
| `PresetMNIST` | gap | (none) | mnist deferred (loader) |
| `PresetRegression` | gap | (none) | requires regression_multi extension |
| `Train` | covered | all actives via `Fit` | — |
| `Query` | covered | xor, binary_classification, iris, regression_sin, regression_multi, persistence, callbacks, style_showcase, higher_order_options, shared_options, precision | — |
| `Verify` | gap | (none) | planned iris/regression_sin extension |
| `AndTrain` | covered | continuation | — |
| `Persistence` (Save / Reload) | covered | persistence | — |

**Gap summary**: 3 elements (`PresetMNIST`, `PresetRegression`, `Verify`) remain uncovered pending mnist/loader work.
