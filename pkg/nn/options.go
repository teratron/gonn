package nn

import (
	"log/slog"
	"slices"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/layer/embedding"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/layer/recurrent"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// Option is the functional-options handle type. Each Option mutates the
// internal Config[T] of an in-construction network. Both the Builder API
// and Options API converge on Config[T] and call the same compile().
//
// AI-Meta:
//   - Purpose: Functional option type for the Options API; applied by New before compile().
//   - Usage: Pass to New[float32](WithInput[float32](4), WithOutput[float32](1, ...)).
//   - Concurrency: Safe.
//   - Related: [New], [MustNew], [WithInput], [WithHiddenLayer], [WithOutput].
//   - Stability: Stable.
type Option[T utils.Float] func(*Config[T])

// New constructs a network using the Functional Options API and returns it
// in Operational state. compile() runs implicitly — there is no separate
// finalisation step.
//
//	nn, err := nn.New[float32](
//	    nn.WithInput[float32](2),
//	    nn.WithHiddenLayer[float32](4, activation.SIGMOID),
//	    nn.WithOutput[float32](1, activation.SIGMOID),
//	    nn.WithLearningRate[float32](0.3),
//	)
//
// AI-Meta:
//   - Purpose: Construct and compile a network in one call using functional options.
//   - Usage: n, err := nn.New[float32](WithInput[float32](4), WithHiddenLayer[float32](8, activation.ReLU), WithOutput[float32](1, activation.SIGMOID)).
//   - Lifecycle: Returns NN directly in Operational state.
//   - Concurrency: SingleGoroutine.
//   - Errors: ErrUserConfig (validation failure, see Compile).
//   - Related: [NewBuilder], [MustNew], [Option].
//   - Stability: Stable.
func New[T utils.Float](opts ...Option[T]) (*NN[T], error) {
	n := &NN[T]{
		Network:    network.New[T](),
		stateField: stateConfiguring,
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt(&n.cfg)
	}
	if err := compile(n, &n.cfg); err != nil {
		return nil, err
	}
	n.stateField = stateOperational
	return n, nil
}

// MustNew is the panic-on-error variant of New, intended for examples and
// tests where a compile failure is a programming bug.
//
// AI-Meta:
//   - Purpose: Panicking wrapper around New; eliminates error handling in contexts where errors are impossible.
//   - Usage: n := nn.MustNew[float32](PresetXOR[float32]()).
//   - Lifecycle: Returns NN in Operational state.
//   - Concurrency: SingleGoroutine.
//   - Related: [New], [NewBuilder], [MustCompile].
//   - Stability: Stable.
func MustNew[T utils.Float](opts ...Option[T]) *NN[T] {
	n, err := New(opts...)
	if err != nil {
		panic("nn.MustNew: " + err.Error())
	}
	return n
}

// ============================================================================
// Topology options
// ============================================================================

// WithInput declares the input-layer size.
//
// AI-Meta:
//   - Purpose: Set the number of input features in the Options API.
//   - Usage: nn.New[float32](WithInput[float32](4), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [NN.Input].
//   - Stability: Stable.
func WithInput[T utils.Float](size uint) Option[T] {
	return func(cfg *Config[T]) {
		cfg.InputSize = size
	}
}

// WithHiddenLayer appends a hidden layer with the supplied size and activation.
// Bias defaults to the DefaultBias set via WithBias; for per-layer bias control
// use the Builder API's Dense method instead.
//
// AI-Meta:
//   - Purpose: Add one hidden layer to the network topology in the Options API.
//   - Usage: nn.New[float32](WithHiddenLayer[float32](8, activation.ReLU), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [NN.Dense], [Sequential].
//   - Stability: Stable.
func WithHiddenLayer[T utils.Float](size uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		cfg.HiddenLayers = append(cfg.HiddenLayers, HiddenLayerSpec[T]{
			Size:       size,
			Activation: act,
			Bias:       cfg.DefaultBias,
		})
	}
}

// WithOutput declares the output-layer size and activation. Bias defaults
// to DefaultBias set via WithBias.
//
// AI-Meta:
//   - Purpose: Declare the output layer size and activation in the Options API.
//   - Usage: nn.New[float32](WithOutput[float32](1, activation.SIGMOID), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [NN.Output].
//   - Stability: Stable.
func WithOutput[T utils.Float](size uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		cfg.OutputSize = size
		cfg.OutputActivation = act
		cfg.OutputBias = cfg.DefaultBias
	}
}

// ============================================================================
// Configuration options
// ============================================================================

// WithLearningRate is the option-form mirror of (*NN[T]).WithLearningRate.
//
// AI-Meta:
//   - Purpose: Set the SGD learning rate in the Options API.
//   - Usage: nn.New[float32](WithLearningRate[float32](0.01), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithLearningRate], [DefaultLearningRate].
//   - Stability: Stable.
func WithLearningRate[T utils.Float](rate T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LearningRate = rate
	}
}

// WithLoss is the option-form mirror of (*NN[T]).WithLoss.
//
// AI-Meta:
//   - Purpose: Set the loss function in the Options API.
//   - Usage: nn.New[float32](WithLoss[float32](loss.MSE), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithLoss], [loss.Type].
//   - Stability: Stable.
func WithLoss[T utils.Float](lossType loss.Type) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LossType = lossType
	}
}

// WithBias sets the global default bias flag for any subsequent
// WithHiddenLayer / WithOutput option. Order matters — options applied
// before WithBias use the prior default.
//
// AI-Meta:
//   - Purpose: Set the default bias flag for subsequently added layers in the Options API.
//   - Usage: nn.New[float32](WithBias[float32](true), WithHiddenLayer[float32](8, activation.ReLU), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithBias], [WithHiddenLayer], [WithOutput].
//   - Stability: Stable.
func WithBias[T utils.Float](use bool) Option[T] {
	return func(cfg *Config[T]) {
		cfg.DefaultBias = use
	}
}

// WithWeightInit is the option-form mirror of (*NN[T]).WithWeightInit.
//
// AI-Meta:
//   - Purpose: Select the weight-init strategy in the Options API.
//   - Usage: nn.New[float32](WithWeightInit[float32](WeightInitHe), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithWeightInit], [WeightInitMethod].
//   - Stability: Stable.
func WithWeightInit[T utils.Float](method WeightInitMethod) Option[T] {
	return func(cfg *Config[T]) {
		cfg.WeightInit = method
	}
}

// WithLossLimit is the option-form mirror of (*NN[T]).WithLossLimit.
//
// AI-Meta:
//   - Purpose: Set the early-stopping loss threshold in the Options API.
//   - Usage: nn.New[float32](WithLossLimit[float32](1e-5), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithLossLimit], [DefaultLossLimit].
//   - Stability: Stable.
func WithLossLimit[T utils.Float](threshold T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LossLimit = threshold
	}
}

// WithMaxIterations is the option-form mirror of (*NN[T]).WithMaxIterations.
//
// AI-Meta:
//   - Purpose: Set the maximum epoch count in the Options API.
//   - Usage: nn.New[float32](WithMaxIterations[float32](5000), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithMaxIterations], [DefaultMaxIterations].
//   - Stability: Stable.
func WithMaxIterations[T utils.Float](count uint) Option[T] {
	return func(cfg *Config[T]) {
		cfg.MaxIterations = count
	}
}

// WithEpochCallback is the option-form mirror of (*NN[T]).WithEpochCallback.
//
// AI-Meta:
//   - Purpose: Register a per-epoch progress callback in the Options API.
//   - Usage: nn.New[float32](WithEpochCallback[float32](func(epoch uint, loss float32) { log.Printf("epoch %d loss %.4f", epoch, loss) }), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithEpochCallback], [WithBatchCallback].
//   - Stability: Stable.
func WithEpochCallback[T utils.Float](fn func(epoch uint, lossValue T)) Option[T] {
	return func(cfg *Config[T]) {
		cfg.EpochCallback = fn
	}
}

// WithBatchCallback is the option-form mirror of (*NN[T]).WithBatchCallback.
//
// AI-Meta:
//   - Purpose: Register a per-batch progress callback in the Options API.
//   - Usage: nn.New[float32](WithBatchCallback[float32](func(batch uint, loss float32) { log.Printf("batch %d loss %.4f", batch, loss) }), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [NN.WithBatchCallback], [WithEpochCallback].
//   - Stability: Stable.
func WithBatchCallback[T utils.Float](fn func(batch uint, lossValue T)) Option[T] {
	return func(cfg *Config[T]) {
		cfg.BatchCallback = fn
	}
}

// WithProfiling enables the optional pprof HTTP endpoint. Pass the listen
// address (e.g. ":6060"); empty string keeps profiling disabled. Errors from
// the listener goroutine are logged at Warn level and never block compile.
//
// AI-Meta:
//   - Purpose: Opt in to pprof profiling at a given address in the Options API.
//   - Usage: nn.New[float32](WithProfiling[float32](":6060"), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New].
//   - Stability: Stable.
func WithProfiling[T utils.Float](addr string) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ProfilingAddr = addr
	}
}

// WithOptimizer replaces the default SGD weight-update rule with opt.
// When not set, compile() falls back to optimizer.DefaultOptimizer(LearningRate).
//
// AI-Meta:
//   - Purpose: Plug in an alternative optimizer (Adam, RMSProp, SGD+Momentum) in the Options API.
//   - Usage: nn.New[float32](WithOptimizer[float32](optimizer.NewAdam[float32](0.001)), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [optimizer.Optimizer].
//   - Stability: Stable.
func WithOptimizer[T utils.Float](opt optimizer.Optimizer[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.Optimizer = opt
	}
}

// WithRegularizer attaches a regularization strategy to the training loop.
// nil disables regularization (the default when option is omitted).
//
// AI-Meta:
//   - Purpose: Attach L1/L2/Dropout or a Compose regularizer in the Options API.
//   - Usage: nn.New[float32](WithRegularizer[float32](regularizer.NewL2[float32](0.01)), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [regularizer.Regularizer].
//   - Stability: Stable.
func WithRegularizer[T utils.Float](reg regularizer.Regularizer[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.Regularizer = reg
	}
}

// WithScheduler attaches a learning-rate scheduler to the training loop.
// The scheduler's Step is called after each epoch (PerEpoch granularity) or
// after each batch (PerStep granularity). nil disables scheduling (default).
// Use optimizer.BindScheduler to wire the scheduler to the active optimizer
// so that Step automatically updates the optimizer's effective rate.
//
// AI-Meta:
//   - Purpose: Attach an LR scheduler (StepLR, WarmUpLR, CosineAnnealingLR, ChainScheduler) in the Options API.
//   - Usage: nn.New[float32](WithScheduler[float32](optimizer.BindScheduler(myOpt, optimizer.NewStepLR[float32](0.1, 10, 0.5))), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [New], [optimizer.Scheduler], [optimizer.BindScheduler].
//   - Stability: Stable.
func WithScheduler[T utils.Float](sched optimizer.Scheduler[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.Scheduler = sched
	}
}

// ============================================================================
// Bulk topology constructors (Track B — l2-deep-builder)
// ============================================================================

// Repeat appends count identical hidden layers (size, act, DefaultBias) to
// the topology. Equivalent to calling WithHiddenLayer count times.
// RepeatCountZero (count == 0) is a no-op here; Compile rejects zero-layer configs.
//
// AI-Meta:
//   - Purpose: Add N identical hidden layers in one Options API call; replaces repeated WithHiddenLayer.
//   - Usage: nn.New[float32](Repeat[float32](100, 256, activation.ReLU), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithHiddenLayer], [Pattern], [WithHiddenLayers], [Sequential].
//   - Stability: Stable.
func Repeat[T utils.Float](count, size uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		for range count {
			cfg.HiddenLayers = append(cfg.HiddenLayers, HiddenLayerSpec[T]{
				Size:       size,
				Activation: act,
				Bias:       cfg.DefaultBias,
			})
		}
	}
}

// Pattern appends block repeated repeats times, producing len(block)×repeats
// hidden layers. A zero-length block or zero repeats is a no-op here;
// Compile enforces the minimum-one-layer rule.
//
// AI-Meta:
//   - Purpose: Add a repeating multi-layer block to the topology in the Options API.
//   - Usage: nn.New[float32](Pattern[float32](block, 33), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [Repeat], [WithHiddenLayers].
//   - Stability: Stable.
func Pattern[T utils.Float](block []HiddenLayerSpec[T], repeats uint) Option[T] {
	return func(cfg *Config[T]) {
		for range repeats {
			cfg.HiddenLayers = append(cfg.HiddenLayers, block...)
		}
	}
}

// WithHiddenLayers appends the given slice of layer specs to the topology
// (append semantics, consistent with WithHiddenLayer singular).
// For replace semantics in the Builder API use (*NN[T]).HiddenLayers.
//
// AI-Meta:
//   - Purpose: Bulk-append a pre-built slice of hidden layer specs in the Options API.
//   - Usage: nn.New[float32](WithHiddenLayers[float32](layers), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [HiddenLayerSpec], [WithHiddenLayer], [Repeat], [Pattern].
//   - Stability: Stable.
func WithHiddenLayers[T utils.Float](layers []HiddenLayerSpec[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.HiddenLayers = append(cfg.HiddenLayers, layers...)
	}
}

// ============================================================================
// Dynamic topology options (Phase 9 — l2-dynamic-topology-impl)
// ============================================================================

// WithTopologyMode opts the network into dynamic topology mutation after
// compile. Pass network.Dynamic to enable AddNeuron / AddHiddenLayer etc.
// The default Immutable keeps the static-topology guarantee of prior phases.
//
// AI-Meta:
//   - Purpose: Opt a compiled network into dynamic topology mutation at construction time.
//   - Usage: nn.New[float32](WithTopologyMode[float32](network.Dynamic), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [network.TopologyMode], [network.Dynamic].
//   - Stability: Stable.
func WithTopologyMode[T utils.Float](mode network.TopologyMode) Option[T] {
	return func(cfg *Config[T]) {
		cfg.TopologyMode = mode
	}
}

// ============================================================================
// Observability options (Phase 9 — l2-logging-strategy, l2-visualization-api)
// ============================================================================

// WithLogger routes NN training lifecycle events to the supplied slog.Logger.
// nil disables per-network structured logging (fallback to utils.Logger).
//
// AI-Meta:
//   - Purpose: Attach a custom slog.Logger for per-network lifecycle events.
//   - Usage: nn.New[float32](WithLogger[float32](slog.New(...)), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [utils.GoLogger], [utils.LevelTrace].
//   - Stability: Stable.
func WithLogger[T utils.Float](l *slog.Logger) Option[T] {
	return func(cfg *Config[T]) {
		cfg.Logger = l
	}
}

// WithVisualizationEndpoint sets the listen address for the optional HTTP
// observability server. An empty string (default) disables the server.
// Example: ":8080" or "127.0.0.1:9000".
//
// AI-Meta:
//   - Purpose: Start the HTTP visualization server at the given address after compile.
//   - Usage: nn.New[float32](WithVisualizationEndpoint[float32](":8080"), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [visualization.VisServer].
//   - Stability: Stable.
func WithVisualizationEndpoint[T utils.Float](addr string) Option[T] {
	return func(cfg *Config[T]) {
		cfg.VisAddr = addr
	}
}

// WithVisualizationToken sets a static bearer token for the visualization
// server. When non-empty, requests must include "Authorization: Bearer <token>".
// An empty token (default) disables authentication.
//
// AI-Meta:
//   - Purpose: Protect the visualization HTTP endpoint with bearer-token auth.
//   - Usage: nn.New[float32](WithVisualizationToken[float32]("my-secret"), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithVisualizationEndpoint].
//   - Stability: Stable.
func WithVisualizationToken[T utils.Float](token string) Option[T] {
	return func(cfg *Config[T]) {
		cfg.VisToken = token
	}
}

// WithVisualizationCORS enables CORS headers on the visualization server so
// browser-based dashboards can connect cross-origin. Default false.
//
// AI-Meta:
//   - Purpose: Enable CORS on the visualization HTTP server for browser dashboards.
//   - Usage: nn.New[float32](WithVisualizationCORS[float32](true), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithVisualizationEndpoint].
//   - Stability: Stable.
func WithVisualizationCORS[T utils.Float](enable bool) Option[T] {
	return func(cfg *Config[T]) {
		cfg.VisCORS = enable
	}
}

// ============================================================================
// Convolutional prefix options (Phase 11 — l2-conv-layers-impl)
// ============================================================================

// WithConv1D appends a 1-D convolutional layer to the prefix stack consumed
// before the Dense hidden chain. compile() rewires the Input layer to the
// final conv-stack output size and initialises the kernel weights via the
// configured WeightInit method.
//
// AI-Meta:
//   - Purpose: Add a Conv1D layer to the preprocessing stack ahead of the Dense head.
//   - Usage: nn.New[float32](WithInput[float32](28*28), WithConv1D[float32](16, 3, 1, conv.PadValid, true), WithFlatten[float32](), WithHiddenLayer[float32](64, activation.ReLU), WithOutput[float32](10, activation.SOFTMAX)).
//   - Concurrency: Safe.
//   - Related: [WithMaxPool1D], [WithAvgPool1D], [WithFlatten], [conv.PadMode].
//   - Stability: Stable.
func WithConv1D[T utils.Float](numFilters, kernelSize, stride int, pad conv.PadMode, useBias bool) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix,
			conv.NewConv1D[T](numFilters, kernelSize, stride, pad, useBias))
	}
}

// WithMaxPool1D appends a 1-D max-pooling layer to the prefix stack.
//
// AI-Meta:
//   - Purpose: Add a MaxPool1D layer to the preprocessing stack.
//   - Usage: nn.New[float32](..., WithConv1D[float32](16, 3, 1, conv.PadValid, true), WithMaxPool1D[float32](2), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv1D], [WithAvgPool1D], [WithFlatten].
//   - Stability: Stable.
func WithMaxPool1D[T utils.Float](poolSize int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewMaxPool1D[T](poolSize))
	}
}

// WithAvgPool1D appends a 1-D average-pooling layer to the prefix stack.
//
// AI-Meta:
//   - Purpose: Add an AvgPool1D layer to the preprocessing stack.
//   - Usage: nn.New[float32](..., WithConv1D[float32](16, 3, 1, conv.PadValid, true), WithAvgPool1D[float32](2), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv1D], [WithMaxPool1D], [WithFlatten].
//   - Stability: Stable.
func WithAvgPool1D[T utils.Float](poolSize int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewAvgPool1D[T](poolSize))
	}
}

// WithFlatten appends a stateless Flatten layer to the prefix stack — used
// to collapse a multi-filter conv output into the 1-D vector consumed by
// the first Dense layer.
//
// AI-Meta:
//   - Purpose: Add a Flatten reshape layer to the preprocessing stack.
//   - Usage: nn.New[float32](..., WithMaxPool1D[float32](2), WithFlatten[float32](), WithHiddenLayer[float32](64, activation.ReLU), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv1D], [WithMaxPool1D], [WithAvgPool1D].
//   - Stability: Stable.
func WithFlatten[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewFlatten[T]())
	}
}

// ============================================================================
// 2-D Convolutional options (Phase 14 — l2-conv-2d-impl)
// ============================================================================

// WithInputShape declares the (channels, height, width) shape of the raw
// network input. Required when a [WithConv2D] / [WithMaxPool2D] /
// [WithAvgPool2D] layer follows that needs to know its CHW dimensions.
// Single-channel square inputs (e.g. flat 784 → (1, 28, 28)) MAY skip this
// option — Conv2D.Forward auto-infers a single-channel square shape via
// integer square root.
//
// AI-Meta:
//   - Purpose: Declare the raw input CHW shape so 2-D conv layers can resolve OutputShape at compile time.
//   - Usage: nn.New[float32](WithInput[float32](3*32*32), WithInputShape[float32](3, 32, 32), WithConv2D[float32](16, 3, 3, 3, 1, 1, conv.PadValid, true), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv2D], [WithMaxPool2D], [WithAvgPool2D], [WithFlatten2D].
//   - Stability: Stable.
func WithInputShape[T utils.Float](channels, height, width int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.InputC = channels
		cfg.InputH = height
		cfg.InputW = width
	}
}

// WithConv2D appends a 2-D convolutional layer to the prefix stack. CHW
// layout per CONV2D-C9. Input must satisfy InChannels*InH*InW = previous
// layer's flat output (or the raw input length declared via WithInput +
// WithInputShape).
//
// AI-Meta:
//   - Purpose: Add a Conv2D layer to the preprocessing stack ahead of the Dense head.
//   - Usage: nn.New[float32](WithInput[float32](28*28), WithConv2D[float32](8, 1, 3, 3, 1, 1, conv.PadValid, true), WithMaxPool2D[float32](2, 2), WithFlatten2D[float32](), WithHiddenLayer[float32](64, activation.ReLU), WithOutput[float32](10, activation.SOFTMAX)).
//   - Concurrency: Safe.
//   - Related: [WithMaxPool2D], [WithAvgPool2D], [WithFlatten2D], [WithInputShape], [conv.PadMode].
//   - Stability: Stable.
func WithConv2D[T utils.Float](numFilters, inChannels, kernelH, kernelW, strideH, strideW int, pad conv.PadMode, useBias bool) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix,
			conv.NewConv2D[T](numFilters, inChannels, kernelH, kernelW, strideH, strideW, pad, useBias))
	}
}

// WithMaxPool2D appends a 2-D max-pooling layer to the prefix stack. Stride
// equals (poolH, poolW) — non-overlapping windows.
//
// AI-Meta:
//   - Purpose: Add a MaxPool2D layer to the preprocessing stack.
//   - Usage: nn.New[float32](..., WithConv2D[float32](8, 1, 3, 3, 1, 1, conv.PadValid, true), WithMaxPool2D[float32](2, 2), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv2D], [WithAvgPool2D], [WithFlatten2D].
//   - Stability: Stable.
func WithMaxPool2D[T utils.Float](poolH, poolW int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewMaxPool2D[T](poolH, poolW))
	}
}

// WithAvgPool2D appends a 2-D average-pooling layer to the prefix stack.
//
// AI-Meta:
//   - Purpose: Add an AvgPool2D layer to the preprocessing stack.
//   - Usage: nn.New[float32](..., WithConv2D[float32](8, 1, 3, 3, 1, 1, conv.PadValid, true), WithAvgPool2D[float32](2, 2), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv2D], [WithMaxPool2D], [WithFlatten2D].
//   - Stability: Stable.
func WithAvgPool2D[T utils.Float](poolH, poolW int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewAvgPool2D[T](poolH, poolW))
	}
}

// WithFlatten2D appends a stateless 2-D Flatten layer to the prefix stack.
// Collapses the CHW feature map into a 1-D vector consumed by the first
// Dense layer; element order follows CONV2D-C9 flat-index formula.
//
// AI-Meta:
//   - Purpose: Add a Flatten2D reshape layer to collapse CHW feature maps into a 1-D vector.
//   - Usage: nn.New[float32](..., WithMaxPool2D[float32](2, 2), WithFlatten2D[float32](), WithHiddenLayer[float32](64, activation.ReLU), ...).
//   - Concurrency: Safe.
//   - Related: [WithConv2D], [WithMaxPool2D], [WithAvgPool2D].
//   - Stability: Stable.
func WithFlatten2D[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewFlatten2D[T]())
	}
}

// ============================================================================
// Normalization options (Phase 10 — l2-normalization-impl)
// ============================================================================

// WithNormAfterLayer inserts a normalizer after the hidden layer at the given
// index. The normalizer is applied to the hidden activations of that layer after
// each forward pass. Multiple calls for the same index overwrite the previous
// normalizer (last-write-wins).
//
// AI-Meta:
//   - Purpose: Register a custom Normalizer to be applied after hidden layer idx during training.
//   - Usage: nn.New[float32](WithNormAfterLayer[float32](0, norm.NewBatchNorm[float32](8)), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithBatchNorm], [WithLayerNorm], [norm.Normalizer].
//   - Stability: Stable.
func WithNormAfterLayer[T utils.Float](idx int, n norm.Normalizer[T]) Option[T] {
	return func(cfg *Config[T]) {
		if cfg.NormLayers == nil {
			cfg.NormLayers = make(map[int]norm.Normalizer[T])
		}
		cfg.NormLayers[idx] = n
	}
}

// WithBatchNorm is a convenience wrapper that attaches a default BatchNorm
// after the hidden layer at idx. The BatchNorm feature count is derived from
// the hidden layer size at Compile time if it was already set, or defaults to
// a placeholder — prefer WithNormAfterLayer for precise control.
// For correctness, call after the corresponding WithHiddenLayer call so the
// layer size is known.
//
// AI-Meta:
//   - Purpose: Convenience shortcut to add BatchNorm after hidden layer idx without specifying features.
//   - Usage: nn.New[float32](WithHiddenLayer[float32](8, activation.ReLU), WithBatchNorm[float32](0), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithNormAfterLayer], [norm.NewBatchNorm].
//   - Stability: Stable.
func WithBatchNorm[T utils.Float](idx int) Option[T] {
	return func(cfg *Config[T]) {
		if idx < len(cfg.HiddenLayers) {
			size := int(cfg.HiddenLayers[idx].Size)
			if size > 0 {
				if cfg.NormLayers == nil {
					cfg.NormLayers = make(map[int]norm.Normalizer[T])
				}
				cfg.NormLayers[idx] = norm.NewBatchNorm[T](size)
				return
			}
		}
		// Layer not defined yet or size=0 — register a sentinel to be resolved at compile.
		if cfg.NormLayers == nil {
			cfg.NormLayers = make(map[int]norm.Normalizer[T])
		}
		cfg.NormLayers[idx] = nil // resolved in compile
	}
}

// WithLayerNorm is a convenience wrapper that attaches a default LayerNorm
// after the hidden layer at idx.
//
// AI-Meta:
//   - Purpose: Convenience shortcut to add LayerNorm after hidden layer idx.
//   - Usage: nn.New[float32](WithHiddenLayer[float32](8, activation.ReLU), WithLayerNorm[float32](0), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [WithNormAfterLayer], [norm.NewLayerNorm].
//   - Stability: Stable.
func WithLayerNorm[T utils.Float](idx int) Option[T] {
	return func(cfg *Config[T]) {
		if idx < len(cfg.HiddenLayers) {
			size := int(cfg.HiddenLayers[idx].Size)
			if size > 0 {
				if cfg.NormLayers == nil {
					cfg.NormLayers = make(map[int]norm.Normalizer[T])
				}
				cfg.NormLayers[idx] = norm.NewLayerNorm[T](size)
				return
			}
		}
		if cfg.NormLayers == nil {
			cfg.NormLayers = make(map[int]norm.Normalizer[T])
		}
		cfg.NormLayers[idx] = nil
	}
}

// ============================================================================
// Callback options (Phase 10 — l2-callbacks-impl)
// ============================================================================

// WithOnIterationEnd registers fn to be called after each epoch's weight update
// (CB-9 boundary atomicity). Multiple calls append to the slice in order (CB-7).
// Returning ErrStopTraining from fn triggers early stopping with best-weight rollback.
//
// AI-Meta:
//   - Purpose: Register a per-epoch callback invoked after weight update completes (CB-9).
//   - Usage: nn.New[float32](WithOnIterationEnd[float32](func(ctx nn.CallbackContext[float32]) error { return nil }), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [CallbackFn], [ErrStopTraining], [WithOnImprovementFound], [WithOnTrainEnd].
//   - Stability: Stable.
func WithOnIterationEnd[T utils.Float](fn CallbackFn[T]) Option[T] {
	return func(cfg *Config[T]) {
		if cfg.Callbacks == nil {
			cfg.Callbacks = &CallbackRegistry[T]{}
		}
		cfg.Callbacks.OnIterationEnd = append(cfg.Callbacks.OnIterationEnd, fn)
	}
}

// WithOnImprovementFound registers fn to be called whenever the epoch loss
// reaches a new minimum (before the OnIterationEnd dispatch for that epoch).
// Multiple calls append in order (CB-7).
//
// AI-Meta:
//   - Purpose: Register a callback triggered on each new loss minimum, ideal for checkpoint saves.
//   - Usage: nn.New[float32](WithOnImprovementFound[float32](saveFn), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [CallbackFn], [ErrStopTraining], [WithOnIterationEnd].
//   - Stability: Stable.
func WithOnImprovementFound[T utils.Float](fn CallbackFn[T]) Option[T] {
	return func(cfg *Config[T]) {
		if cfg.Callbacks == nil {
			cfg.Callbacks = &CallbackRegistry[T]{}
		}
		cfg.Callbacks.OnImprovementFound = append(cfg.Callbacks.OnImprovementFound, fn)
	}
}

// WithMetaLearner attaches an inner network that tunes the outer network's
// registered hyperparameters at each training iteration. ml must be fully
// constructed (inner *NN[T] Operational, params registered) before Compile.
// The option is rejected with ErrMetaLearnerRunning if the outer network is
// already training at compile time.
//
// AI-Meta:
//   - Purpose: Attach a MetaLearner to the outer NN for recursive self-optimization.
//   - Usage: nn.New[float32](WithMetaLearner[float32](ml), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [MetaLearner], [ParamAccessor].
//   - Stability: Stable.
func WithMetaLearner[T utils.Float](ml *MetaLearner[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.MetaLearner = ml
	}
}

// WithOnTrainEnd registers fn to be called when Fit returns, regardless of
// how training ended (normal completion, ErrStopTraining, error, or panic).
// The CallbackContext.StopReason field is set (CB-8 guarantee).
//
// AI-Meta:
//   - Purpose: Register a finalisation callback that fires on any Fit exit path (CB-8).
//   - Usage: nn.New[float32](WithOnTrainEnd[float32](logResultFn), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [CallbackFn], [StopReason], [WithOnIterationEnd].
//   - Stability: Stable.
func WithOnTrainEnd[T utils.Float](fn CallbackFn[T]) Option[T] {
	return func(cfg *Config[T]) {
		if cfg.Callbacks == nil {
			cfg.Callbacks = &CallbackRegistry[T]{}
		}
		cfg.Callbacks.OnTrainEnd = append(cfg.Callbacks.OnTrainEnd, fn)
	}
}

// ============================================================================
// Recurrent prefix options (Phase 16 — l2-recurrent-impl REC-9)
// ============================================================================

// WithSimpleRNN appends an Elman SimpleRNN layer to the prefix stack. The
// layer satisfies [conv.Layer] and is initialised by compile() via the shared
// weight-sampler path alongside convolutional kernels (REC-9 composition).
//
// AI-Meta:
//   - Purpose: Add a SimpleRNN layer to the preprocessing stack ahead of the Dense head.
//   - Usage: nn.New[float64](WithInput[float64](seqLen*inSize), WithSimpleRNN[float64](seqLen, inSize, hidden), WithLastStep[float64](seqLen, hidden), WithHiddenLayer[float64](8, activation.ReLU), WithOutput[float64](1, activation.SIGMOID)).
//   - Concurrency: Safe.
//   - Related: [WithGRU], [WithLSTM], [WithLastStep], [recurrent.SimpleRNN].
//   - Stability: Stable.
func WithSimpleRNN[T utils.Float](seqLen, inSize, hidden int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, recurrent.NewSimpleRNN[T](seqLen, inSize, hidden))
	}
}

// WithLSTM appends an LSTM layer to the prefix stack. Forget-gate bias is
// initialised to 1.0 per REC-3 convention during compile().
//
// AI-Meta:
//   - Purpose: Add an LSTM layer to the preprocessing stack ahead of the Dense head.
//   - Usage: nn.New[float64](WithInput[float64](seqLen*inSize), WithLSTM[float64](seqLen, inSize, hidden), WithLastStep[float64](seqLen, hidden), WithHiddenLayer[float64](8, activation.ReLU), WithOutput[float64](1, activation.SIGMOID)).
//   - Concurrency: Safe.
//   - Related: [WithGRU], [WithSimpleRNN], [WithLastStep], [recurrent.LSTM].
//   - Stability: Stable.
func WithLSTM[T utils.Float](seqLen, inSize, hidden int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, recurrent.NewLSTM[T](seqLen, inSize, hidden))
	}
}

// WithGRU appends a GRU (Gated Recurrent Unit) layer to the prefix stack.
// The 3-gate fused matmul (reset, update, candidate) is weight-initialised
// by compile() via the shared Init path (REC-9).
//
// AI-Meta:
//   - Purpose: Add a GRU layer to the preprocessing stack ahead of the Dense head.
//   - Usage: nn.New[float64](WithInput[float64](seqLen*inSize), WithGRU[float64](seqLen, inSize, hidden), WithLastStep[float64](seqLen, hidden), WithHiddenLayer[float64](8, activation.ReLU), WithOutput[float64](1, activation.SIGMOID)).
//   - Concurrency: Safe.
//   - Related: [WithLSTM], [WithSimpleRNN], [WithLastStep], [recurrent.GRU].
//   - Stability: Stable.
func WithGRU[T utils.Float](seqLen, inSize, hidden int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, recurrent.NewGRU[T](seqLen, inSize, hidden))
	}
}

// WithLastStep appends a stateless LastStep collapser to the prefix stack.
// It reduces the flat sequence output [seqLen*hidden] to the final-timestep
// vector [hidden] before the Dense head (REC-8).
//
// AI-Meta:
//   - Purpose: Add a LastStep sequence-to-vector collapser to the preprocessing stack.
//   - Usage: nn.New[float64](..., WithGRU[float64](seqLen, inSize, hidden), WithLastStep[float64](seqLen, hidden), ...).
//   - Concurrency: Safe.
//   - Related: [WithGRU], [WithLSTM], [WithSimpleRNN], [recurrent.LastStep].
//   - Stability: Stable.
func WithLastStep[T utils.Float](seqLen, hidden int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, recurrent.NewLastStep[T](seqLen, hidden))
	}
}

// ============================================================================
// Compute backend options (Phase 16 — l2-backend-gpu §6 phase D)
// ============================================================================

// WithBackend sets the compute backend used for Dense layer kernels. When the
// supplied backend returns [utils.ErrBackendUnavailable] on compile-time probe,
// compile() logs a Warn and substitutes the always-available CPU reference
// backend (l1-compute-backend §5.3 graceful fallback). nil is equivalent to
// omitting this option — both result in the CPU backend.
//
// AI-Meta:
//   - Purpose: Select a compute backend (CPU, OpenCL, CUDA) at construction time; unavailable backends fall back to CPU.
//   - Usage: nn.New[float32](WithBackend[float32](myGPUBackend), ...).
//   - Concurrency: Safe.
//   - Errors: Never surfaces ErrBackendUnavailable to the caller; compile() absorbs it with a Warn log.
//   - Related: [Option], [compute.Backend], [utils.ErrBackendUnavailable].
//   - Stability: Stable.
func WithBackend[T utils.Float](b compute.Backend[T]) Option[T] {
	return func(cfg *Config[T]) {
		cfg.Backend = b
	}
}

// ============================================================================
// Gradient clipping (Phase 16 — l2-recurrent-impl REC-7)
// ============================================================================

// WithGradClipNorm enables gradient clipping by global L2 norm. Before each
// optimizer step, [optimizer.ClipByGlobalNorm] scales all gradient slices so
// that their global L2 norm does not exceed threshold. A threshold ≤ 0
// disables clipping (the default).
//
// AI-Meta:
//   - Purpose: Clip gradients in-place before the optimizer step to stabilise BPTT (REC-7).
//   - Usage: nn.New[float32](WithGradClipNorm[float32](1.0), ...).
//   - Concurrency: Safe.
//   - Related: [Option], [optimizer.ClipByGlobalNorm], [WithSimpleRNN], [WithLSTM], [WithGRU].
//   - Stability: Stable.
func WithGradClipNorm[T utils.Float](threshold T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.GradClipNorm = threshold
	}
}

// ============================================================================
// Attention options (Phase 17 — l1-attention ATT-1..ATT-10)
// ============================================================================

// WithAttention appends a single-head scaled dot-product attention layer to
// the prefix stack. Equivalent to WithMultiHeadAttention with numHeads=1.
// Use WithCausalAttention after this call to enable causal masking.
//
// AI-Meta:
//   - Purpose: Add a single-head attention layer (NumHeads=1) to the preprocessing stack.
//   - Usage: nn.New[float64](WithInput[float64](seqLen*dmodel), WithAttention[float64](seqLen, dmodel), WithOutput[float64](numClasses, activation.SOFTMAX)).
//   - Concurrency: Safe.
//   - Related: [WithMultiHeadAttention], [WithCausalAttention], [attention.MultiHeadAttention].
//   - Stability: Stable.
func WithAttention[T utils.Float](seqLen, dmodel int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, attention.NewAttention[T](seqLen, dmodel, false))
	}
}

// WithMultiHeadAttention appends a multi-head attention layer to the prefix
// stack. panics if dmodel is not divisible by numHeads.
// Use WithCausalAttention after this call to enable causal masking.
//
// AI-Meta:
//   - Purpose: Add a multi-head attention layer to the preprocessing stack (ATT-1..ATT-10).
//   - Usage: nn.New[float64](WithInput[float64](seqLen*dmodel), WithMultiHeadAttention[float64](seqLen, dmodel, numHeads), WithOutput[float64](numClasses, activation.SOFTMAX)).
//   - Concurrency: Safe.
//   - Related: [WithAttention], [WithCausalAttention], [attention.MultiHeadAttention].
//   - Stability: Stable.
func WithMultiHeadAttention[T utils.Float](seqLen, dmodel, numHeads int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix,
			attention.NewMultiHeadAttention[T](seqLen, dmodel, numHeads, false))
	}
}

// WithCausalAttention sets Causal=true on the most recently added
// [attention.MultiHeadAttention] layer in the prefix stack. It is a no-op
// when no attention layer has been added yet.
//
// AI-Meta:
//   - Purpose: Enable causal (autoregressive) masking on the last attention layer (ATT-5).
//   - Usage: nn.New[float64](WithMultiHeadAttention[float64](seqLen, dmodel, numHeads), WithCausalAttention[float64](), ...).
//   - Concurrency: Safe.
//   - Related: [WithAttention], [WithMultiHeadAttention], [attention.MultiHeadAttention.Causal].
//   - Stability: Stable.
func WithCausalAttention[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		for _, v := range slices.Backward(cfg.ConvPrefix) {
			if mha, ok := v.(*attention.MultiHeadAttention[T]); ok {
				mha.Causal = true
				return
			}
		}
	}
}

// ============================================================================
// Embedding options (Phase 17 — l1-embedding EMB-1..EMB-10)
// ============================================================================

// WithEmbeddingStack appends an [embedding.EmbeddingStack] to the prefix stack.
// It combines a learnable token lookup table with a positional encoding layer
// (sinusoidal by default) into a single IDLayer. Accepts integer token IDs
// via the Forward(x []T) path (IDs encoded as float values).
//
// AI-Meta:
//   - Purpose: Add a combined token+positional embedding layer (EMB-1..EMB-10) to the prefix stack.
//   - Usage: nn.New[float64](WithInput[float64](seqLen), WithEmbeddingStack[float64](vocabSize, seqLen, dmodel, embedding.Sinusoidal), WithMultiHeadAttention[float64](seqLen, dmodel, numHeads), WithOutput[float64](numClasses, activation.SOFTMAX)).
//   - Concurrency: Safe.
//   - Related: [WithTokenEmbedding], [embedding.EmbeddingStack], [embedding.PositionalMode].
//   - Stability: Stable.
func WithEmbeddingStack[T utils.Float](vocabSize, seqLen, dmodel int, mode embedding.PositionalMode) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix,
			embedding.NewEmbeddingStack[T](vocabSize, seqLen, dmodel, mode))
	}
}

// WithTokenEmbedding appends a bare [embedding.TokenEmbedding] (no positional
// encoding) to the prefix stack. Use when positional information is provided
// by another mechanism or is not required.
//
// AI-Meta:
//   - Purpose: Add a learnable token lookup table without positional encoding (EMB-1..EMB-4, EMB-9).
//   - Usage: nn.New[float64](WithInput[float64](seqLen), WithTokenEmbedding[float64](vocabSize, seqLen, dmodel), ...).
//   - Concurrency: Safe.
//   - Related: [WithEmbeddingStack], [embedding.TokenEmbedding].
//   - Stability: Stable.
func WithTokenEmbedding[T utils.Float](vocabSize, seqLen, dmodel int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix,
			embedding.NewTokenEmbedding[T](vocabSize, seqLen, dmodel))
	}
}
