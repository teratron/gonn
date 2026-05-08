package nn

import (
	"github.com/teratron/gonn/pkg/activation"
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
	n, err := New[T](opts...)
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

