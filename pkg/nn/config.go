// Package nn — public facade for the GoNN library.
//
// This file owns the internal configuration state populated by both the
// Builder API ([builder.go]) and the Functional Options API ([options.go]).
// Per [l2-nn-facade] §5.4 "Convergence point" both styles share this
// Config type and the same compile() finalisation routine — there is no
// duplicated logic between Style A and Style B.
package nn

import (
	"log/slog"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/compute"
	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/optimizer"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// DefaultExcessiveLayersWarning is the hidden-layer count at which compile
// emits a soft warning for ExcessiveLayers (l2-deep-builder §5.7).
const DefaultExcessiveLayersWarning = 10_000

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
// String-typed so JSON-serialised Config payloads stay self-describing
// across library versions without locking in numeric IDs.
//
// AI-Meta:
//   - Purpose: Closed string enum for selecting the weight-init strategy applied at Compile.
//   - Usage: Pass WeightInitXavier / WeightInitHe / WeightInitRandom to WithWeightInit.
//   - Concurrency: Safe.
//   - Related: [WeightInitXavier], [WeightInitHe], [WeightInitRandom], [WithWeightInit].
//   - Stability: Stable.
type WeightInitMethod string

const (
	// WeightInitXavier draws weights from Glorot uniform U[-a, a],
	// a = sqrt(6 / (fanIn + fanOut)). Recommended for tanh / sigmoid layers.
	// The default applied by Compile when WeightInit is unset.
	//
	// AI-Meta:
	//   - Purpose: Glorot uniform weight initialiser; optimal for tanh/sigmoid activations.
	//   - Usage: Pass WeightInitXavier to WithWeightInit or set Config.WeightInit directly.
	//   - Concurrency: Safe.
	//   - Related: [WeightInitMethod], [WeightInitHe], [WeightInitRandom], [WithWeightInit].
	//   - Stability: Stable.
	WeightInitXavier WeightInitMethod = "xavier"

	// WeightInitHe draws weights from He normal N(0, sigma^2),
	// sigma = sqrt(2 / fanIn). Recommended for ReLU / LeakyReLU layers.
	//
	// AI-Meta:
	//   - Purpose: He normal weight initialiser; optimal for ReLU/LeakyReLU activations.
	//   - Usage: Pass WeightInitHe to WithWeightInit for ReLU-based topologies.
	//   - Concurrency: Safe.
	//   - Related: [WeightInitMethod], [WeightInitXavier], [WeightInitRandom], [WithWeightInit].
	//   - Stability: Stable.
	WeightInitHe WeightInitMethod = "he"

	// WeightInitRandom draws weights from uniform U[-1, 1). Soft-warns
	// at Compile when paired with deep stacks (>5 hidden layers) due
	// to gradient-explosion risk.
	//
	// AI-Meta:
	//   - Purpose: Uniform random weight initialiser; use only for shallow or experimental networks.
	//   - Usage: Pass WeightInitRandom to WithWeightInit for quick experiments.
	//   - Concurrency: Safe.
	//   - Related: [WeightInitMethod], [WeightInitXavier], [WeightInitHe], [WithWeightInit].
	//   - Stability: Stable.
	WeightInitRandom WeightInitMethod = "random"
)

// Defaults applied by Compile when the corresponding Config field is left at
// its zero value. Public so tests and examples can reference the canonical
// values without hard-coding them.
//
// AI-Meta:
//   - Purpose: Canonical baseline values for hyperparameters; applied by applyDefaults before compile.
//   - Usage: Reference in assertions or option chains that need to override then restore defaults.
//   - Concurrency: Safe.
//   - Related: [Config], [Compile].
//   - Stability: Stable.
const (
	DefaultLearningRate     = 0.3
	DefaultMaxIterations    = uint(10_000)
	DefaultLossLimit        = 1e-4
	DefaultWeightInitMethod = WeightInitXavier
	DefaultLossMode         = loss.MSE
)

// HiddenLayerSpec captures the shape of one hidden layer. A slice of these
// inside Config[T] lets higher-order options like Sequential and DeepNetwork
// compose multiple hidden layers from a single call.
//
// AI-Meta:
//   - Purpose: Per-layer shape descriptor for one element of the multi-hidden chain.
//   - Usage: Populated by Dense / Hidden builder methods or WithHiddenLayer; stored in Config.HiddenLayers.
//   - Concurrency: NotSafe.
//   - Related: [Config], [Dense], [WithHiddenLayer], [Sequential], [DeepNetwork].
//   - Stability: Stable.
type HiddenLayerSpec[T utils.Float] struct {
	Size       uint
	Activation activation.Type
	Bias       bool
}

// Config is the staging buffer that both the Builder API and Functional
// Options API write into before Compile reads it. The generic parameter T
// flows through every public surface in this package.
//
// AI-Meta:
//   - Purpose: Unified configuration struct shared by both construction styles; frozen by Compile.
//   - Usage: Populated via builder chain or option functions; passed to compile() internally.
//   - Lifecycle: Populated in Configuring state; read-only after Compile transitions NN to Operational.
//   - Concurrency: NotSafe; mutated by builder/option methods, read by compile().
//   - Related: [HiddenLayerSpec], [Compile], [Option], [NN.Config].
//   - Stability: Stable.
type Config[T utils.Float] struct {
	Backend          compute.Backend[T]
	LossLimit        T
	Optimizer        optimizer.Optimizer[T]
	Regularizer      regularizer.Regularizer[T]
	Scheduler        optimizer.Scheduler[T]
	GradClipNorm     T
	LearningRate     T
	Callbacks        *CallbackRegistry[T]
	BatchCallback    func(batch uint, lossValue T)
	EpochCallback    func(epoch uint, lossValue T)
	MetaLearner      *MetaLearner[T]
	Logger           *slog.Logger
	NormLayers       map[int]norm.Normalizer[T]
	VisAddr          string
	WeightInit       WeightInitMethod
	WeightInitSeed   uint64
	VisToken         string
	ProfilingAddr    string
	HiddenLayers     []HiddenLayerSpec[T]
	ConvPrefix       []conv.Layer[T]
	MaxIterations    uint
	InputH           int
	InputW           int
	InputC           int
	InputSize        uint
	OutputSize       uint
	LossType         loss.Type
	OutputActivation activation.Type
	TopologyMode     network.TopologyMode
	VisCORS          bool
	DefaultBias      bool
	OutputBias       bool
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
