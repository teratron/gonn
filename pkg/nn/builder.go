package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// ============================================================================
// Topology methods
// ============================================================================

// Input declares the input layer size. Order convention: call Input
// before any Dense / Hidden / Output. Calling Input twice overwrites
// the previous value with a Logger.Debug trace — useful for repl-like
// reconfiguration before Compile.
func (n *NN[T]) Input(size uint) *NN[T] {
	if !n.guardConfiguring("Input") {
		return n
	}
	if n.cfg.InputSize != 0 {
		utils.Logger.Debug("Input size overwritten",
			"old", n.cfg.InputSize, "new", size)
	}
	n.cfg.InputSize = size
	return n
}

// Dense appends a hidden layer with the given size, activation, and
// bias setting. The activation symbol must be registered in the
// activation dispatcher — invalid symbols surface at Compile() as
// ErrUserConfig (UnknownActivation).
func (n *NN[T]) Dense(size uint, act activation.Type, bias bool) *NN[T] {
	if !n.guardConfiguring("Dense") {
		return n
	}
	n.cfg.HiddenLayers = append(n.cfg.HiddenLayers, HiddenLayerSpec[T]{
		Size:       size,
		Activation: act,
		Bias:       bias,
	})
	return n
}

// Hidden is a documented alias for Dense — kept for naming-convention
// users who think in terms of "input/hidden/output".
func (n *NN[T]) Hidden(size uint, act activation.Type, bias bool) *NN[T] {
	return n.Dense(size, act, bias)
}

// Output declares the output layer. Note the breaking change from v1.0:
// loss is no longer a parameter here — set it via [WithLoss]. Rationale
// per [l2-nn-facade] §5.2: structural concerns (size, activation, bias)
// and training concerns (loss) live on separate axes.
func (n *NN[T]) Output(size uint, act activation.Type, bias bool) *NN[T] {
	if !n.guardConfiguring("Output") {
		return n
	}
	if n.cfg.OutputSize != 0 {
		utils.Logger.Debug("Output size overwritten",
			"old", n.cfg.OutputSize, "new", size)
	}
	n.cfg.OutputSize = size
	n.cfg.OutputActivation = act
	n.cfg.OutputBias = bias
	return n
}

// ============================================================================
// Configuration methods
// ============================================================================

// WithLearningRate sets the SGD learning rate applied during Train.
// Non-positive values are accepted at the chain level but rejected by
// Compile() with ErrUserConfig (RateNonPositive).
func (n *NN[T]) WithLearningRate(rate T) *NN[T] {
	if !n.guardConfiguring("WithLearningRate") {
		return n
	}
	n.cfg.LearningRate = rate
	return n
}

// WithLoss sets the loss function symbol consumed by the training loop.
// Default at Compile() is loss.MSE.
func (n *NN[T]) WithLoss(lossType loss.Type) *NN[T] {
	if !n.guardConfiguring("WithLoss") {
		return n
	}
	n.cfg.LossType = lossType
	return n
}

// WithBias sets the global default for any Dense / Output layer added
// AFTER this call. Layers added before WithBias keep their per-call
// bias setting — this method does not retroactively rewrite history.
func (n *NN[T]) WithBias(use bool) *NN[T] {
	if !n.guardConfiguring("WithBias") {
		return n
	}
	n.cfg.DefaultBias = use
	return n
}

// WithWeightInit selects the weight-initialisation strategy. Unknown
// methods are accepted at the chain level but rejected by Compile()
// with ErrUserConfig (UnknownInit).
func (n *NN[T]) WithWeightInit(method WeightInitMethod) *NN[T] {
	if !n.guardConfiguring("WithWeightInit") {
		return n
	}
	n.cfg.WeightInit = method
	return n
}

// WithLossLimit sets the early-stopping threshold (per [l1-training-semantics]).
// Train returns once the running loss drops below this value. Default
// at Compile() is DefaultLossLimit.
func (n *NN[T]) WithLossLimit(threshold T) *NN[T] {
	if !n.guardConfiguring("WithLossLimit") {
		return n
	}
	n.cfg.LossLimit = threshold
	return n
}

// WithMaxIterations bounds the training loop. Default at Compile() is
// DefaultMaxIterations.
func (n *NN[T]) WithMaxIterations(count uint) *NN[T] {
	if !n.guardConfiguring("WithMaxIterations") {
		return n
	}
	n.cfg.MaxIterations = count
	return n
}

// ============================================================================
// Callback methods
// ============================================================================

// WithEpochCallback registers a callback invoked at the end of every
// training epoch. Synchronous — a long-running callback blocks training.
// Pass nil to disable a previously-set callback.
func (n *NN[T]) WithEpochCallback(fn func(epoch uint, lossValue T)) *NN[T] {
	if !n.guardConfiguring("WithEpochCallback") {
		return n
	}
	n.cfg.EpochCallback = fn
	return n
}

// WithBatchCallback registers a callback invoked at the end of every
// training batch. Synchronous; same caveats as [WithEpochCallback].
func (n *NN[T]) WithBatchCallback(fn func(batch uint, lossValue T)) *NN[T] {
	if !n.guardConfiguring("WithBatchCallback") {
		return n
	}
	n.cfg.BatchCallback = fn
	return n
}

// ============================================================================
// Finalisation
// ============================================================================

// Compile validates the staged configuration, applies defaults, builds
// the underlying network, and transitions the *NN[T] to Operational
// state. Returns the same *NN[T] (for chain-style examples) plus an
// error if validation fails — every error wraps utils.ErrUserConfig
// per the project taxonomy (C32).
//
// Compile is idempotent for the post-compile case: a second call on an
// already-Operational network returns (n, ErrAlreadyCompiled) without
// re-running validation. Mutating builder methods after Compile emit
// Logger.Warn and become no-ops.
func (n *NN[T]) Compile() (*NN[T], error) {
	switch n.stateField {
	case stateOperational:
		return n, utils.Newf(utils.ErrUserConfig, "Compile: network already compiled (idempotent: returning existing instance)")
	case stateUninitialized:
		return nil, utils.Newf(utils.ErrUserConfig, "Compile: network is uninitialised — use NewBuilder() or New(opts...)")
	}
	if err := compile(n, &n.cfg); err != nil {
		return nil, err
	}
	n.stateField = stateOperational
	return n, nil
}

// MustCompile is the panic-on-error variant of Compile, reserved for
// examples and tests where any compile error is a programming bug.
func (n *NN[T]) MustCompile() *NN[T] {
	out, err := n.Compile()
	if err != nil {
		panic("nn.MustCompile: " + err.Error())
	}
	return out
}
