package nn

import (
	"log/slog"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/layer/norm"
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
//   - Concurrency: SingleGoroutine during construction; ReadSafe for Query after return.
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
//   - Concurrency: SingleGoroutine during construction.
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
//   - Related: [WithConv1D], [WithAvgPool1D], [WithFlatten].
//   - Stability: Stable.
func WithMaxPool1D[T utils.Float](poolSize int) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewMaxPool1D[T](poolSize))
	}
}

// WithAvgPool1D appends a 1-D average-pooling layer to the prefix stack.
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
//   - Related: [WithConv1D], [WithMaxPool1D], [WithAvgPool1D].
//   - Stability: Stable.
func WithFlatten[T utils.Float]() Option[T] {
	return func(cfg *Config[T]) {
		cfg.ConvPrefix = append(cfg.ConvPrefix, conv.NewFlatten[T]())
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
