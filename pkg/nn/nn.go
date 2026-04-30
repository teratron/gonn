package nn

import (
	"sync/atomic"

	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

// NN is the public facade type. It embeds network.Network[T] (composition,
// not inheritance — INV-7) and adds the construction lifecycle plus the
// training-control state cell.
//
// Both fluent styles return *NN[T]:
//   - Builder: [NewBuilder] starts the chain in Configuring state.
//   - Options: [New] / [MustNew] run compile() implicitly and return an
//     Operational network.
//
// Concurrent reads of a compiled network are safe (e.g. parallel Query).
// Train must be invoked from a single goroutine — it owns the internal
// PCG and the weights.
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
}

// NewBuilder is the entry point for the Builder API (Style A). Returns
// an *NN[T] in Configuring state with default-initialised configuration —
// callers chain Input/Dense/Output/WithX calls then finalise via Compile.
//
//	nn, err := nn.NewBuilder[float32]().
//	    Input(2).
//	    Dense(4, activation.SIGMOID, true).
//	    Output(1, activation.SIGMOID, true).
//	    WithLearningRate(0.3).
//	    WithLoss(loss.MSE).
//	    Compile()
func NewBuilder[T utils.Float]() *NN[T] {
	n := &NN[T]{
		Network:    network.New[T](),
		stateField: stateConfiguring,
	}
	utils.Logger.Debug("NN builder initialised", "state", n.stateField.String())
	return n
}

// State reports the current lifecycle position. Exposed for advanced
// callers and for the post-compile guard helpers; ordinary code rarely
// needs to inspect it.
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

// Config returns a copy of the staged configuration. Useful for tests
// that want to assert the chain populated the right fields, and for
// future persistence-spec round-trips.
func (n *NN[T]) Config() Config[T] {
	// Shallow copy — HiddenLayers slice header is duplicated but the
	// backing array is shared. Callers must not mutate the returned
	// slice; this is the same contract as Network.Cells().
	return n.cfg
}
