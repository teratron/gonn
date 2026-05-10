package nn

import (
	"sync/atomic"

	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// NN is the public facade type. It embeds network.Network[T] by composition
// and adds the construction lifecycle and the training-control state machine.
//
// Two fluent construction styles are available:
//   - Builder API: [NewBuilder] → chain methods → [Compile].
//   - Options API: [New] / [MustNew] run compile implicitly and return Operational.
//
// Concurrent reads of an Operational network are safe (parallel Query).
// Train/Fit must run from a single goroutine — they own the weights.
//
// AI-Meta:
//   - Purpose: Public network handle for topology configuration, training, and inference.
//   - Usage: Configure via NewBuilder or New, call Train/Fit for learning, Query for inference.
//   - Lifecycle: Configuring (after NewBuilder) → Operational (after Compile / New / MustNew).
//   - Concurrency: ReadSafe for Query after Compile; Train and Fit require SingleGoroutine.
//   - Related: [NewBuilder], [New], [MustNew], [network.Network].
//   - Constraints: Topology is immutable after Compile; weights must not be mutated concurrently.
//   - Stability: Stable.
type NN[T utils.Float] struct {
	network.Network[T] `json:"network" xml:"network"`

	// Internal staging buffer for both styles.
	cfg Config[T]

	// Construction-lifecycle position. Mutated only from the goroutine
	// that calls Compile() / NewBuilder() / New() — read by post-compile
	// guard helpers.
	stateField state

	// Training-control state cell. Atomic so Pause / Resume / Stop can
	// be called from a different goroutine while Train is running. See
	// [control.go] for the public surface.
	control atomic.Int32

	// opt is the resolved optimizer (always non-nil after compile).
	opt optimizer.Optimizer[T]

	// reg is the optional regularizer (nil = no regularization).
	reg regularizer.Regularizer[T]

	// sched is the optional learning-rate scheduler (nil = no scheduling).
	sched optimizer.Scheduler[T]

	// weightBuf / gradBuf are reused per training step to avoid
	// per-sample allocations in the hot training loop.
	weightBuf []T
	gradBuf   []T
}

// NewBuilder is the entry point for the Builder API. Returns an *NN[T] in
// Configuring state; callers chain topology and hyperparameter methods then
// call Compile to transition to Operational.
//
//	nn, err := nn.NewBuilder[float32]().
//	    Input(2).
//	    Dense(4, activation.SIGMOID, true).
//	    Output(1, activation.SIGMOID, true).
//	    WithLoss(loss.MSE).
//	    Compile()
//
// AI-Meta:
//   - Purpose: Start a fluent Builder chain for configuring a new network.
//   - Usage: nn.NewBuilder[float32]().Input(N).Dense(M, ...).Output(K, ...).Compile().
//   - Lifecycle: Returns NN in Configuring state; must call Compile before Train/Query.
//   - Concurrency: SingleGoroutine; must complete Compile before sharing across goroutines.
//   - Related: [NN], [Compile], [New], [MustNew].
//   - Stability: Stable.
func NewBuilder[T utils.Float]() *NN[T] {
	n := &NN[T]{
		Network:    network.New[T](),
		stateField: stateConfiguring,
	}
	utils.Logger.Debug("NN builder initialised", "state", n.stateField.String())
	return n
}

// State reports the current lifecycle position (Uninitialized, Configuring,
// or Operational). Ordinary code rarely needs to inspect this — it is
// primarily for advanced callers and test assertions.
//
// AI-Meta:
//   - Purpose: Expose the construction-lifecycle state for introspection and assertions.
//   - Concurrency: ReadSafe; field is written only during construction.
//   - Related: [NN], [NewBuilder], [Compile].
//   - Stability: Stable.
func (n *NN[T]) State() state {
	return n.stateField
}

// guardConfiguring is invoked by every builder/option mutation method
// before it touches Config. When called on an Operational network it
// emits a Logger.Warn (per L1 INV-2 — immutable topology) and returns
// false to signal that the caller should bail out without mutating.
//
// Returns true when the network is in Configuring state and the
// mutation is legal.
func (n *NN[T]) guardConfiguring(method string) bool {
	switch n.stateField {
	case stateConfiguring:
		return true
	case stateOperational:
		utils.Logger.Warn(
			"network already compiled — mutation ignored",
			"method", method,
			"state", n.stateField.String(),
		)
		return false
	default:
		utils.Logger.Warn(
			"builder method called on uninitialised network — use NewBuilder() or New()",
			"method", method,
			"state", n.stateField.String(),
		)
		return false
	}
}

// Config returns a shallow copy of the staged configuration. Useful for
// tests that want to assert the chain populated the right fields. Callers
// must not mutate the returned HiddenLayers slice.
//
// AI-Meta:
//   - Purpose: Read-only introspection of the staged or compiled configuration.
//   - Concurrency: ReadSafe; returns a shallow copy of the internal Config.
//   - Related: [Config], [Compile].
//   - Stability: Stable.
func (n *NN[T]) Config() Config[T] {
	// Shallow copy — HiddenLayers slice header is duplicated but the
	// backing array is shared. Callers must not mutate the returned
	// slice; this is the same contract as Network.Cells().
	return n.cfg
}
