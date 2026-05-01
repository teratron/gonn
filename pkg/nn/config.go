// Package nn — public facade for the GoNN library.
//
// This file owns the internal configuration state populated by both the
// Builder API ([builder.go]) and the Functional Options API ([options.go]).
// Per [l2-nn-facade] §5.4 "Convergence point" both styles share this
// Config type and the same compile() finalisation routine — there is no
// duplicated logic between Style A and Style B.
package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// state enumerates the construction-lifecycle positions of an *NN[T].
// Per [l2-nn-facade] §5.1 the legal call set differs by state — illegal
// calls are no-ops with a Logger.Warn, never panics.
type state uint8

const (
	stateUninitialized state = iota
	stateConfiguring
	stateOperational
)

// String returns a human-readable form of the state — used in
// Logger.Warn messages emitted by post-Compile builder/option calls.
func (s state) String() string {
	switch s {
	case stateUninitialized:
		return "Uninitialized"
	case stateConfiguring:
		return "Configuring"
	case stateOperational:
		return "Operational"
	default:
		return "unknown"
	}
}

// WeightInitMethod identifies a recognised weight-initialisation strategy.
// String-typed (not numeric) so that JSON-serialised Config[T] payloads
// remain self-describing — this matches the planned persistence contract
// in [l1-network-persistence] §5.2 without locking in numeric IDs.
type WeightInitMethod string

const (
	// WeightInitXavier draws weights from Glorot uniform U[-a, a],
	// a = sqrt(6 / (fanIn + fanOut)). Recommended for tanh / sigmoid
	// layers. The default applied by Compile() when WeightInit is unset.
	WeightInitXavier WeightInitMethod = "xavier"

	// WeightInitHe draws weights from He normal N(0, sigma^2),
	// sigma = sqrt(2 / fanIn). Recommended for ReLU / LeakyReLU layers.
	WeightInitHe WeightInitMethod = "he"

	// WeightInitRandom draws weights from uniform U[-1, 1). Soft-warns
	// at Compile() when paired with deep stacks (>5 hidden layers) due
	// to gradient-explosion risk.
	WeightInitRandom WeightInitMethod = "random"
)

// Defaults applied by Compile() when the corresponding Config field is
// left at its zero value. Public so tests / examples can reference them
// without re-deriving the numbers from the spec text.
const (
	DefaultLearningRate     = 0.3
	DefaultMaxIterations    = uint(10_000)
	DefaultLossLimit        = 1e-4
	DefaultWeightInitMethod = WeightInitXavier
	DefaultLossMode         = loss.MSE
)

// HiddenLayerSpec captures the shape of one hidden layer. A slice of
// these inside Config[T] allows higher-order options like Sequential and
// DeepNetwork to compose multiple hidden layers from a single call.
type HiddenLayerSpec[T utils.Float] struct {
	Size       uint
	Activation activation.Type
	Bias       bool
}

// Config is the internal staging buffer populated by builder methods and
// functional options. It is intentionally unexported in v2.0 — promotion
// to a public, serialisable type is reserved for the persistence spec
// (see [l2-nn-facade] §5.5 forward-extension hook). The generic parameter
// T flows through every public surface in this package.
type Config[T utils.Float] struct {
	InputSize        uint
	HiddenLayers     []HiddenLayerSpec[T]
	OutputSize       uint
	OutputActivation activation.Type
	OutputBias       bool

	LearningRate  T
	LossType      loss.Type
	LossLimit     T
	MaxIterations uint
	WeightInit    WeightInitMethod
	DefaultBias   bool

	EpochCallback func(epoch uint, lossValue T)
	BatchCallback func(batch uint, lossValue T)

	// ProfilingAddr enables the optional pprof HTTP listener
	// (PERF-5). Empty (zero value) keeps the listener disabled.
	ProfilingAddr string
}

// applyDefaults fills any zero-valued fields with the Defaults constants.
// Called by compile() exactly once, immediately before validation, so
// downstream code can rely on every field being normalised.
//
// Note: LossType has zero value loss.MSE (the iota-zero entry in the
// dispatcher), so an unset LossType already defaults to MSE without
// special handling — that is the intended spec behaviour.
func (c *Config[T]) applyDefaults() {
	if c.LearningRate <= 0 {
		c.LearningRate = T(DefaultLearningRate)
	}
	if c.LossLimit <= 0 {
		c.LossLimit = T(DefaultLossLimit)
	}
	if c.MaxIterations == 0 {
		c.MaxIterations = DefaultMaxIterations
	}
	if c.WeightInit == "" {
		c.WeightInit = DefaultWeightInitMethod
	}
}
