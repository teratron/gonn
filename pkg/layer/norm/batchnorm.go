package norm

import (
	"encoding/json"
	"sync/atomic"

	"github.com/teratron/gonn/pkg/utils"
)

// ErrBatchNormSingleSample is a warning sentinel used when BatchNorm.Forward is
// called in NormTrain mode with a 1-element input (variance is zero by definition,
// producing a degenerate normalized value of 0). Forward returns the input
// unchanged and logs this condition. Callers processing single-feature layers
// should switch to LayerNorm.
//
// AI-Meta:
//   - Purpose: Sentinel documenting single-feature degenerate case for BatchNorm in NormTrain mode.
//   - Usage: Check utils.Logger output; Forward falls back to identity and logs this.
//   - Related: [BatchNorm], [LayerNorm].
//   - Stability: Stable.
var ErrBatchNormSingleSample = utils.Newf(utils.ErrUserConfig,
	"BatchNorm.Forward: single-sample batch in NormTrain mode (len=1) — variance is 0; use LayerNorm for single-feature layers")

// BatchNormOption is a functional option for NewBatchNorm.
//
// AI-Meta:
//   - Purpose: Functional option type for BatchNorm construction.
//   - Usage: NewBatchNorm[float32](4, WithBatchNormEps[float32](1e-3)).
//   - Related: [NewBatchNorm], [WithBatchNormEps], [WithBatchNormMomentum], [WithBatchNormAffine].
//   - Stability: Stable.
type BatchNormOption[T utils.Float] func(*BatchNorm[T])

// WithBatchNormEps sets the epsilon value added to variance before computing
// the standard deviation. Must be positive. Defaults to 1e-5.
//
// AI-Meta:
//   - Purpose: Override the variance-floor epsilon for numerical stability in BatchNorm.
//   - Related: [BatchNormOption], [NewBatchNorm].
//   - Stability: Stable.
func WithBatchNormEps[T utils.Float](eps T) BatchNormOption[T] {
	return func(b *BatchNorm[T]) { b.eps = eps }
}

// WithBatchNormMomentum sets the EMA momentum for running stats updates.
// Must be in (0, 1]. Defaults to 0.1.
//
// AI-Meta:
//   - Purpose: Override the exponential moving average factor for running statistics.
//   - Related: [BatchNormOption], [NewBatchNorm].
//   - Stability: Stable.
func WithBatchNormMomentum[T utils.Float](m T) BatchNormOption[T] {
	return func(b *BatchNorm[T]) { b.momentum = m }
}

// WithBatchNormAffine controls whether learnable affine parameters (gamma, beta)
// are allocated. When false, GradSlots returns (nil, nil). Defaults to true.
//
// AI-Meta:
//   - Purpose: Disable per-feature scale/shift when a downstream affine layer is present.
//   - Related: [BatchNormOption], [NewBatchNorm], [BatchNorm.GradSlots].
//   - Stability: Stable.
func WithBatchNormAffine[T utils.Float](a bool) BatchNormOption[T] {
	return func(b *BatchNorm[T]) { b.affine = a }
}

// BatchNorm is a batch normalization layer that normalizes its input by
// computing mean and variance over the feature vector. In NormTrain mode
// the statistics are computed from the current input and used to update
// the exponential moving averages (runningMean, runningVar). In NormEval
// mode the frozen running statistics are applied instead (NORM-6).
//
// Affine parameters gamma (scale) and beta (shift) are per-feature and
// participate in the optimizer cycle via GradSlots (NORM-7). Initialised
// as gamma=1, beta=0 (identity transform at construction).
//
// AI-Meta:
//   - Purpose: Normalization layer using input mean/variance with EMA stat tracking.
//   - Usage: bn := norm.NewBatchNorm[float32](features); out := bn.Forward(x).
//   - Lifecycle: SetMode(NormTrain) before training epochs; SetMode(NormEval) for inference.
//   - Concurrency: SetMode is goroutine-safe; Forward must be called from a single goroutine.
//   - Errors: Logs ErrBatchNormSingleSample when len(x)==1 in NormTrain; returns x unchanged.
//   - Related: [NewBatchNorm], [Normalizer], [LayerNorm], [GroupNorm].
//   - Stability: Stable.
type BatchNorm[T utils.Float] struct {
	features    int
	eps         T
	momentum    T
	affine      bool
	gamma       []T
	beta        []T
	gammaGrad   []T
	betaGrad    []T
	runningMean T
	runningVar  T
	mode        atomic.Int32
}

// NewBatchNorm constructs a BatchNorm layer for the given feature count.
// Default: eps=1e-5, momentum=0.1, affine=true with gamma=1 and beta=0.
//
// AI-Meta:
//   - Purpose: Construct a BatchNorm with per-feature affine params and zero running stats.
//   - Usage: bn := norm.NewBatchNorm[float32](hiddenSize).
//   - Errors: Panics if features < 1.
//   - Related: [BatchNorm], [BatchNormOption].
//   - Stability: Stable.
func NewBatchNorm[T utils.Float](features int, opts ...BatchNormOption[T]) *BatchNorm[T] {
	if features < 1 {
		panic("norm.NewBatchNorm: features must be >= 1")
	}
	b := &BatchNorm[T]{
		features: features,
		eps:      T(1e-5),
		momentum: T(0.1),
		affine:   true,
		// runningMean starts at 0, runningVar starts at 1 (identity variance).
		runningMean: 0,
		runningVar:  1,
	}
	for _, opt := range opts {
		opt(b)
	}
	if b.affine {
		b.gamma = make([]T, features)
		b.beta = make([]T, features)
		b.gammaGrad = make([]T, features)
		b.betaGrad = make([]T, features)
		for i := range b.gamma {
			b.gamma[i] = 1 // identity scale
		}
		// beta is already zero from make
	}
	return b
}

// Forward normalizes x and returns the transformed output of the same length (NORM-1).
// In NormTrain mode: computes batch mean/variance over x, applies EMA update to
// running statistics, and normalizes using batch statistics. In NormEval mode:
// uses the frozen running statistics without update.
// If len(x) == 1 in NormTrain the variance is degenerate (=0); returns x unchanged
// and logs ErrBatchNormSingleSample.
//
// AI-Meta:
//   - Purpose: Apply batch normalization to a single feature vector, updating running stats in train mode.
//   - Concurrency: NotSafe; must not be called concurrently.
//   - Errors: Logs ErrBatchNormSingleSample on degenerate single-feature input in NormTrain mode.
//   - Related: [BatchNorm], [Normalizer.Forward].
//   - Stability: Stable.
func (b *BatchNorm[T]) Forward(x []T) []T {
	if len(x) == 0 {
		return x
	}

	var mean, variance T

	if NormMode(b.mode.Load()) == NormTrain {
		if len(x) < 2 {
			// Degenerate single-feature batch: variance=0, normalized value=0.
			// Fall back to identity and log the issue.
			utils.Logger.Warn(ErrBatchNormSingleSample.Error())
			return x
		}
		mean, variance = batchStats(x)
		// Exponential moving average update.
		b.runningMean = (1-b.momentum)*b.runningMean + b.momentum*mean
		b.runningVar = (1-b.momentum)*b.runningVar + b.momentum*variance
	} else {
		mean = b.runningMean
		variance = b.runningVar
	}

	sd := stddev(variance, b.eps)
	xHat := make([]T, len(x))
	for i, v := range x {
		xHat[i] = (v - mean) / sd
	}
	return applyAffine(xHat, b.gamma, b.beta)
}

// SetMode switches between NormTrain (0) and NormEval (1). Goroutine-safe via
// atomic store so pkg/nn.SetTrain / SetEval can call it from any goroutine.
//
// AI-Meta:
//   - Purpose: Atomically switch operational mode; NormTrain uses batch stats, NormEval uses running stats.
//   - Concurrency: Safe.
//   - Related: [BatchNorm], [NormMode], [Normalizer.SetMode].
//   - Stability: Stable.
func (b *BatchNorm[T]) SetMode(m NormMode) {
	b.mode.Store(int32(m))
}

// GradSlots returns the gradient accumulation slices for gamma and beta.
// Returns (nil, nil) when affine is disabled (NORM-7). The training loop
// passes these to the optimizer step alongside Dense weight gradients.
//
// AI-Meta:
//   - Purpose: Expose per-feature gradient slots for optimizer integration (NORM-7).
//   - Usage: gammaGrad, betaGrad := bn.GradSlots(); optimizer.Step(gammaGrad, betaGrad).
//   - Related: [BatchNorm], [Normalizer.GradSlots].
//   - Stability: Stable.
func (b *BatchNorm[T]) GradSlots() (gamma, beta []T) {
	if !b.affine {
		return nil, nil
	}
	return b.gammaGrad, b.betaGrad
}

// InputSize returns the expected input feature count.
//
// AI-Meta:
//   - Purpose: Report input dimensionality for topology integration (NORM-8).
//   - Related: [BatchNorm], [Normalizer.InputSize].
//   - Stability: Stable.
func (b *BatchNorm[T]) InputSize() int { return b.features }

// OutputSize always equals InputSize (NORM-1 shape preservation).
//
// AI-Meta:
//   - Purpose: Report output dimensionality; always equals InputSize for BatchNorm.
//   - Related: [BatchNorm], [Normalizer.OutputSize].
//   - Stability: Stable.
func (b *BatchNorm[T]) OutputSize() int { return b.features }

// MarshalJSON encodes the BatchNorm state for persistence (NORM-9).
// Encodes features, eps, momentum, affine, gamma, beta, running_mean,
// running_var, and mode.
//
// AI-Meta:
//   - Purpose: JSON serialization for checkpoint and persistence round-trip (NORM-9).
//   - Related: [BatchNorm], [UnmarshalJSON].
//   - Stability: Stable.
func (b *BatchNorm[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Features    int   `json:"features"`
		Eps         T     `json:"eps"`
		Momentum    T     `json:"momentum"`
		Affine      bool  `json:"affine"`
		Gamma       []T   `json:"gamma,omitempty"`
		Beta        []T   `json:"beta,omitempty"`
		RunningMean T     `json:"running_mean"`
		RunningVar  T     `json:"running_var"`
		Mode        int32 `json:"mode"`
	}{
		Features:    b.features,
		Eps:         b.eps,
		Momentum:    b.momentum,
		Affine:      b.affine,
		Gamma:       b.gamma,
		Beta:        b.beta,
		RunningMean: b.runningMean,
		RunningVar:  b.runningVar,
		Mode:        b.mode.Load(),
	})
}

// UnmarshalJSON restores BatchNorm state from a JSON payload (NORM-9).
//
// AI-Meta:
//   - Purpose: JSON deserialization for checkpoint and persistence round-trip (NORM-9).
//   - Related: [BatchNorm], [MarshalJSON].
//   - Stability: Stable.
func (b *BatchNorm[T]) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Features    int   `json:"features"`
		Eps         T     `json:"eps"`
		Momentum    T     `json:"momentum"`
		Affine      bool  `json:"affine"`
		Gamma       []T   `json:"gamma,omitempty"`
		Beta        []T   `json:"beta,omitempty"`
		RunningMean T     `json:"running_mean"`
		RunningVar  T     `json:"running_var"`
		Mode        int32 `json:"mode"`
	}{}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	b.features = aux.Features
	b.eps = aux.Eps
	b.momentum = aux.Momentum
	b.affine = aux.Affine
	b.gamma = aux.Gamma
	b.beta = aux.Beta
	b.gammaGrad = make([]T, aux.Features)
	b.betaGrad = make([]T, aux.Features)
	b.runningMean = aux.RunningMean
	b.runningVar = aux.RunningVar
	b.mode.Store(aux.Mode)
	return nil
}

// batchStats computes the arithmetic mean and population variance of x in one pass.
func batchStats[T utils.Float](x []T) (mean, variance T) {
	n := T(len(x))
	for _, v := range x {
		mean += v
	}
	mean /= n
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= n
	return mean, variance
}
