# GoNN Public API Reference

Condensed signatures and descriptions for AI context efficiency.
All types parameterized by `T utils.Float` (`float32 | float64`).

## Package `pkg/nn`

### Types

| Type | Description |
| :--- | :--- |
| `NN[T]` | Public network handle; configure → compile → train/query |
| `Config[T]` | Staging buffer shared by Builder and Options APIs |
| `HiddenLayerSpec[T]` | `{Size uint, Activation activation.Type, Bias bool}` |
| `Option[T]` | `func(*Config[T])` — functional option type |
| `Sample[T]` | `{Input []T, Target []T}` — one training sample |
| `WeightInitMethod` | String enum: `"xavier"` / `"he"` / `"random"` |

### Construction

| Symbol | Signature | Description |
| :--- | :--- | :--- |
| `NewBuilder[T]` | `() *NN[T]` | Start a Builder chain (Configuring state) |
| `New[T]` | `(opts ...Option[T]) (*NN[T], error)` | Options API — compiles implicitly |
| `MustNew[T]` | `(opts ...Option[T]) *NN[T]` | Panic variant of New |

### Builder Methods (Style A)

| Method | Signature | Description |
| :--- | :--- | :--- |
| `Input` | `(size uint) *NN[T]` | Declare input layer size |
| `Dense` | `(size uint, act activation.Type, bias bool) *NN[T]` | Add one hidden layer |
| `Hidden` | same as Dense | Alias for Dense |
| `Repeat` | `(count, size uint, act activation.Type, bias bool) *NN[T]` | Add N identical hidden layers |
| `Pattern` | `(block []HiddenLayerSpec[T], repeats uint) *NN[T]` | Repeat a multi-layer block |
| `HiddenLayers` | `(layers []HiddenLayerSpec[T]) *NN[T]` | Replace all hidden layers |
| `Output` | `(size uint, act activation.Type, bias bool) *NN[T]` | Declare output layer |
| `WithLoss` | `(loss.Type) *NN[T]` | Set loss function |
| `WithLearningRate` | `(T) *NN[T]` | Set learning rate |
| `WithBias` | `(bool) *NN[T]` | Set default bias flag |
| `WithWeightInit` | `(WeightInitMethod) *NN[T]` | Set weight-init strategy |
| `WithLossLimit` | `(T) *NN[T]` | Early-stopping threshold |
| `WithMaxIterations` | `(uint) *NN[T]` | Maximum epoch count |
| `WithOptimizer` | `(optimizer.Optimizer[T]) *NN[T]` | Replace SGD default |
| `WithRegularizer` | `(regularizer.Regularizer[T]) *NN[T]` | Attach regularizer |
| `WithScheduler` | `(optimizer.Scheduler[T]) *NN[T]` | Attach LR scheduler |
| `WithEpochCallback` | `(func(uint, T)) *NN[T]` | Per-epoch progress hook |
| `WithBatchCallback` | `(func(uint, T)) *NN[T]` | Per-batch progress hook |
| `Compile` | `() (*NN[T], error)` | Finalize; transition to Operational |
| `MustCompile` | `() *NN[T]` | Panic variant of Compile |

### Option Constructors (Style B)

| Function | Signature | Description |
| :--- | :--- | :--- |
| `WithInput[T]` | `(uint) Option[T]` | Set input size |
| `WithHiddenLayer[T]` | `(uint, activation.Type) Option[T]` | Add one hidden layer (DefaultBias) |
| `WithHiddenLayers[T]` | `([]HiddenLayerSpec[T]) Option[T]` | Append a slice of layers |
| `Repeat[T]` | `(count, size uint, act activation.Type) Option[T]` | Add N identical layers (DefaultBias) |
| `Pattern[T]` | `([]HiddenLayerSpec[T], uint) Option[T]` | Repeat a block |
| `WithOutput[T]` | `(uint, activation.Type) Option[T]` | Set output layer |
| `WithLoss[T]` | `(loss.Type) Option[T]` | Set loss function |
| `WithLearningRate[T]` | `(T) Option[T]` | Set learning rate |
| `WithBias[T]` | `(bool) Option[T]` | Set default bias |
| `WithWeightInit[T]` | `(WeightInitMethod) Option[T]` | Set weight-init |
| `WithOptimizer[T]` | `(optimizer.Optimizer[T]) Option[T]` | Replace optimizer |
| `WithRegularizer[T]` | `(regularizer.Regularizer[T]) Option[T]` | Attach regularizer |
| `WithScheduler[T]` | `(optimizer.Scheduler[T]) Option[T]` | Attach LR scheduler |
| `WithLossLimit[T]` | `(T) Option[T]` | Early-stopping threshold |
| `WithMaxIterations[T]` | `(uint) Option[T]` | Maximum epochs |
| `WithEpochCallback[T]` | `(func(uint, T)) Option[T]` | Epoch callback |
| `WithBatchCallback[T]` | `(func(uint, T)) Option[T]` | Batch callback |

### Higher-Order Options and Presets

| Function | Description |
| :--- | :--- |
| `Sequential[T](count, size uint, act)` | N identical layers using DefaultBias |
| `DeepNetwork[T](startSize, layers uint, act)` | Pyramid with halving size |
| `StandardSetup[T](rate T)` | SGD + MSE + Xavier + DefaultBias |
| `PresetXOR[T]()` | 2→SIGMOID(4)→SIGMOID(1), MSE, Xavier |
| `PresetMNIST[T]()` | 784→ReLU(128)→ReLU(64)→Softmax(10), CCE, He |
| `PresetRegression[T](inputSize, hiddenSize)` | Pyramid ReLU, MSE |

### Training Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `Fit` | `([]Sample[T]) (epochs uint, loss T, err error)` | Managed multi-epoch loop |
| `Train` | `(input, target []T) (T, error)` | Single forward+backward+update |
| `Query` | `([]T) ([]T, error)` | Inference (no weight update) |
| `Verify` | `([]Sample[T]) (T, error)` | Compute mean loss without update |

### Defaults

| Constant | Value |
| :--- | :--- |
| `DefaultLearningRate` | `0.3` |
| `DefaultMaxIterations` | `10_000` |
| `DefaultLossLimit` | `1e-4` |
| `DefaultWeightInitMethod` | `WeightInitXavier` |
| `DefaultLossMode` | `loss.MSE` |

## Package `pkg/optimizer`

| Symbol | Description |
| :--- | :--- |
| `Optimizer[T]` | Interface: `Step(w, d []T) error`, `Reset()`, `LearningRate() T`, `SaveState/LoadState` |
| `LearningRateSetter[T]` | Optional interface: `SetLearningRate(T)` — all built-ins implement it |
| `Scheduler[T]` | Interface: `Step() T`, `Reset()`, `Granularity()`, `SaveState/LoadState` |
| `Granularity` | `PerEpoch` (default) / `PerStep` |
| `NewSGD[T](lr)` | Vanilla SGD |
| `NewAdam[T](lr)` | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| `NewRMSProp[T](lr)` | RMSProp (α=0.99, ε=1e-8) |
| `NewSGDMomentum[T](lr)` | SGD + momentum (γ=0.9) |
| `NewStepLR[T](lr0, stepSize, gamma)` | Step-decay scheduler |
| `NewWarmUpLR[T](lr0, warmupSteps)` | Linear warm-up (PerStep default) |
| `NewCosineAnnealingLR[T](lr0, lrMin, tMax)` | Cosine annealing |
| `NewChainScheduler[T](segments)` | Sequential multi-scheduler |
| `BindScheduler[T](opt, sched)` | Wire scheduler to optimizer |

## Package `pkg/regularizer`

| Symbol | Description |
| :--- | :--- |
| `Regularizer[T]` | Interface: `Penalty([]T) T`, `ApplyMask([]T, training bool) []T` |
| `NewL1[T](lambda)` | L1 penalty |
| `NewL2[T](lambda)` | L2 penalty |
| `NewDropout[T](rate)` | Dropout mask (training only) |
| `Compose[T](regs...)` | Combine multiple regularizers |
