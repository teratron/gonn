package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

// Option is the functional-options handle. Each Option mutates the
// internal Config[T] of an in-construction network. Per [l2-nn-facade]
// §5.3 — both styles converge on Config[T] and call the same compile().
type Option[T utils.Float] func(*Config[T])

// New constructs a network using the Functional Options API and returns
// it in Operational state. New runs compile() implicitly: there is no
// separate finalisation step in Style B.
//
//	nn, err := nn.New[float32](
//	    nn.WithInput[float32](2),
//	    nn.WithHiddenLayer[float32](4, activation.SIGMOID),
//	    nn.WithOutput[float32](1, activation.SIGMOID),
//	    nn.WithLearningRate[float32](0.3),
//	)
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

// MustNew is the panic-on-error variant of New, reserved for examples
// and tests where compile errors are programming bugs.
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
func WithInput[T utils.Float](size uint) Option[T] {
	return func(cfg *Config[T]) {
		cfg.InputSize = size
	}
}

// WithHiddenLayer appends a hidden layer with the supplied size and
// activation. Bias defaults to the global DefaultBias set via WithBias —
// callers that need per-layer bias control should fall back to the
// Builder API where Dense() takes bias as a positional argument.
func WithHiddenLayer[T utils.Float](size uint, act activation.Type) Option[T] {
	return func(cfg *Config[T]) {
		cfg.HiddenLayers = append(cfg.HiddenLayers, HiddenLayerSpec[T]{
			Size:       size,
			Activation: act,
			Bias:       cfg.DefaultBias,
		})
	}
}

// WithOutput declares the output-layer size and activation. Bias
// defaults to DefaultBias.
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
func WithLearningRate[T utils.Float](rate T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LearningRate = rate
	}
}

// WithLoss is the option-form mirror of (*NN[T]).WithLoss.
func WithLoss[T utils.Float](lossType loss.Type) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LossType = lossType
	}
}

// WithBias sets the global default bias flag for any subsequent
// WithHiddenLayer / WithOutput option in the same chain. Order
// matters — options applied before WithBias use the previous default.
func WithBias[T utils.Float](use bool) Option[T] {
	return func(cfg *Config[T]) {
		cfg.DefaultBias = use
	}
}

// WithWeightInit is the option-form mirror of (*NN[T]).WithWeightInit.
func WithWeightInit[T utils.Float](method WeightInitMethod) Option[T] {
	return func(cfg *Config[T]) {
		cfg.WeightInit = method
	}
}

// WithLossLimit is the option-form mirror of (*NN[T]).WithLossLimit.
func WithLossLimit[T utils.Float](threshold T) Option[T] {
	return func(cfg *Config[T]) {
		cfg.LossLimit = threshold
	}
}

// WithMaxIterations is the option-form mirror of (*NN[T]).WithMaxIterations.
func WithMaxIterations[T utils.Float](count uint) Option[T] {
	return func(cfg *Config[T]) {
		cfg.MaxIterations = count
	}
}

// WithEpochCallback is the option-form mirror of (*NN[T]).WithEpochCallback.
func WithEpochCallback[T utils.Float](fn func(epoch uint, lossValue T)) Option[T] {
	return func(cfg *Config[T]) {
		cfg.EpochCallback = fn
	}
}

// WithBatchCallback is the option-form mirror of (*NN[T]).WithBatchCallback.
func WithBatchCallback[T utils.Float](fn func(batch uint, lossValue T)) Option[T] {
	return func(cfg *Config[T]) {
		cfg.BatchCallback = fn
	}
}

// WithProfiling enables the optional net/http/pprof endpoint per
// [l2-perf-impl] §5.4 (PERF-5). The argument is the listen address
// passed to http.ListenAndServe (e.g. ":6060"). Empty addr keeps
// profiling disabled — the same as not calling the option at all.
//
// The pprof handlers are registered on http.DefaultServeMux via the
// blank-imported net/http/pprof package (see profiling.go); the goroutine
// that runs ListenAndServe is fire-and-forget — listener errors are
// logged at Warn level and never block compile.
func WithProfiling[T utils.Float](addr string) Option[T] {
	return func(cfg *Config[T]) {
		cfg.ProfilingAddr = addr
	}
}
