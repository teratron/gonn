package nn

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
	"github.com/teratron/gonn/pkg/visualization"
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
//   - Concurrency: ReadSafe.
//   - Related: [NewBuilder], [New], [MustNew], [network.Network].
//   - Constraints: Topology is immutable after Compile; weights must not be mutated concurrently.
//   - Stability: Stable.
type NN[T utils.Float] struct {
	cfg                Config[T]
	opt                optimizer.Optimizer[T]
	sched              optimizer.Scheduler[T]
	reg                regularizer.Regularizer[T]
	vis                *visualization.VisServer
	callbacks          *CallbackRegistry[T]
	normLayers         map[int]norm.Normalizer[T]
	backend            compute.Backend[T] // CPU by default; set by WithBackend; nil impossible after compile
	convPrefix         []conv.Layer[T]
	convBuf            []T
	convGradBuf        []T
	gradBuf            []T
	weightBuf          []T
	network.Network[T] `json:"network" xml:"network"`
	rawInputSize       uint
	control            atomic.Int32
	stateField         state
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

// SetTrain switches all registered normalization layers to NormTrain mode
// so they use batch statistics and update running averages during Forward.
// Call before each training epoch to ensure correct BatchNorm behaviour.
//
// AI-Meta:
//   - Purpose: Propagate NormTrain mode to all Normalizer instances before training (NORM-6).
//   - Usage: n.SetTrain(); for _, s := range dataset { n.Train(s.Input, s.Target) }.
//   - Concurrency: NotSafe; must not overlap with Forward calls.
//   - Related: [SetEval], [norm.NormTrain], [norm.Normalizer].
//   - Stability: Stable.
func (n *NN[T]) SetTrain() {
	for _, nl := range n.normLayers {
		if nl != nil {
			nl.SetMode(norm.NormTrain)
		}
	}
}

// SetEval switches all registered normalization layers to NormEval mode so
// they use frozen running statistics during Forward. Call before inference
// to ensure BatchNorm does not mutate running stats.
//
// AI-Meta:
//   - Purpose: Propagate NormEval mode to all Normalizer instances before inference (NORM-6).
//   - Usage: n.SetEval(); output, err := n.Query(input).
//   - Concurrency: NotSafe; must not overlap with Forward calls.
//   - Related: [SetTrain], [norm.NormEval], [norm.Normalizer].
//   - Stability: Stable.
func (n *NN[T]) SetEval() {
	for _, nl := range n.normLayers {
		if nl != nil {
			nl.SetMode(norm.NormEval)
		}
	}
}

// Close stops the optional visualization HTTP server and releases its port.
// Safe to call on networks without a visualization server (no-op). Returns
// the first error encountered during shutdown; the shutdown timeout is 5 s.
//
// AI-Meta:
//   - Purpose: Release resources held by the optional visualization server.
//   - Concurrency: Safe; can be called from any goroutine after Compile.
//   - Related: [WithVisualizationEndpoint], [visualization.VisServer].
//   - Stability: Stable.
func (n *NN[T]) Close() error {
	if n.vis == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return n.vis.Stop(ctx)
}

// TopologyVersion returns the monotonic counter incremented by every successful
// topology mutation on the embedded network. Zero for immutable networks.
//
// AI-Meta:
//   - Purpose: Expose topology change counter for checkpoint invalidation and observability.
//   - Concurrency: Safe; delegates to atomic load on Network[T].
//   - Related: [network.Network.TopologyVersion], [WithTopologyMode].
//   - Stability: Stable.
func (n *NN[T]) TopologyVersion() uint64 {
	return n.Network.TopologyVersion()
}
