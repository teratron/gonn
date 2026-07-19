package norm

import (
	"encoding/json"
	"sync/atomic"

	"github.com/teratron/gonn/pkg/utils"
)

// ErrBatchNormSingleSample is retained for API compatibility. The historical
// implementation normalized ACROSS the feature vector (LayerNorm semantics
// with scalar running stats) and degenerated on single-feature layers; the
// per-feature rework below no longer has that failure mode, so this sentinel
// is never logged by Forward anymore.
//
// AI-Meta:
//   - Purpose: Legacy sentinel from the pre-v0.11 whole-vector implementation; no longer emitted.
//   - Related: [BatchNorm], [LayerNorm].
//   - Stability: Deprecated.
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

// BatchNorm is a batch-normalization layer with PER-FEATURE running
// statistics maintained as exponential moving averages over the sample
// stream (the GoNN training loop feeds one sample at a time, so there is
// no mini-batch axis to average over — "sample-stream EMA" semantics).
//
// In NormTrain mode each Forward normalizes feature i with the CURRENT
// running mean/variance (constants with respect to this sample — this is
// what makes the analytic gradient exact) and then folds the sample into
// the EMA. In NormEval mode the frozen statistics are applied without
// update (NORM-6).
//
// The pre-v0.11 implementation normalized across the feature vector with
// scalar running stats — conceptually LayerNorm with an EMA bolted on —
// and was replaced wholesale (audit B1).
//
// Affine parameters gamma (scale) and beta (shift) are per-feature and
// participate in training via GradSlots / ApplyGradSGD (NORM-7).
// Initialised as gamma=1, beta=0 (identity transform at construction).
//
// AI-Meta:
//   - Purpose: Per-feature normalization with sample-stream EMA statistics.
//   - Usage: bn := norm.NewBatchNorm[float32](features); out := bn.Forward(x).
//   - Lifecycle: SetMode(NormTrain) before training epochs; SetMode(NormEval) for inference.
//   - Concurrency: SetMode is goroutine-safe; Forward/Backward must be called from a single goroutine; ForwardInference is ReadSafe.
//   - Related: [NewBatchNorm], [Normalizer], [LayerNorm], [GroupNorm].
//   - Stability: Stable.
type BatchNorm[T utils.Float] struct {
	eps         T
	momentum    T
	runningMean []T
	runningVar  []T
	gamma       []T
	beta        []T
	gammaGrad   []T
	betaGrad    []T
	// xHat and invSd are cached by Forward for Backward.
	xHat     []T
	invSd    []T
	features int
	mode     atomic.Int32
	affine   bool
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
		// Per-feature running stats: mean starts at 0, variance at 1
		// (identity transform before any sample has been observed).
		runningMean: make([]T, features),
		runningVar:  make([]T, features),
		xHat:        make([]T, features),
		invSd:       make([]T, features),
	}
	for i := range b.runningVar {
		b.runningVar[i] = 1
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

// Forward normalizes each feature of x with its per-feature running
// statistics and returns the transformed output of the same length (NORM-1).
//
// NormTrain: feature i is normalized with the running stats AS THEY WERE
// BEFORE this sample (constants w.r.t. x, so Backward's gradient is exact),
// then the sample is folded into the EMA:
//
//	runningMean_i ← (1−m)·runningMean_i + m·x_i
//	runningVar_i  ← (1−m)·runningVar_i  + m·(x_i − runningMean_i^old)²
//
// NormEval: frozen running statistics, no update.
//
// AI-Meta:
//   - Purpose: Per-feature normalization with sample-stream EMA update in train mode.
//   - Concurrency: NotSafe; mutates running stats and caches. Use ForwardInference for concurrent reads.
//   - Related: [BatchNorm], [Backward], [ForwardInference], [Normalizer.Forward].
//   - Stability: Stable.
func (b *BatchNorm[T]) Forward(x []T) []T {
	if len(x) == 0 {
		return x
	}
	n := min(len(x), b.features)
	train := NormMode(b.mode.Load()) == NormTrain
	if cap(b.xHat) < n {
		b.xHat = make([]T, n)
		b.invSd = make([]T, n)
	} else {
		b.xHat = b.xHat[:n]
		b.invSd = b.invSd[:n]
	}
	for i := range n {
		sd := stddev(b.runningVar[i], b.eps)
		b.xHat[i] = (x[i] - b.runningMean[i]) / sd
		b.invSd[i] = 1 / sd
		if train {
			d := x[i] - b.runningMean[i]
			b.runningMean[i] = (1-b.momentum)*b.runningMean[i] + b.momentum*x[i]
			b.runningVar[i] = (1-b.momentum)*b.runningVar[i] + b.momentum*d*d
		}
	}
	return applyAffine(b.xHat, b.gamma, b.beta)
}

// ForwardInference computes the eval-mode output without mutating any layer
// state — no EMA updates, no caches — so concurrent inference goroutines can
// share one instance under a read lock.
//
// AI-Meta:
//   - Purpose: Pure, mutation-free forward for concurrent inference; always uses running stats.
//   - Concurrency: ReadSafe (assuming no concurrent Forward, which the facade's RWMutex guarantees).
//   - Related: [BatchNorm], [Forward], [Normalizer.ForwardInference].
//   - Stability: Stable.
func (b *BatchNorm[T]) ForwardInference(x []T) []T {
	if len(x) == 0 {
		return x
	}
	n := min(len(x), b.features)
	xHat := make([]T, n)
	for i := range n {
		sd := stddev(b.runningVar[i], b.eps)
		xHat[i] = (x[i] - b.runningMean[i]) / sd
	}
	return applyAffine(xHat, b.gamma, b.beta)
}

// Backward computes ∂L/∂x from the upstream gradient ∂L/∂y and accumulates
// ∂L/∂γ and ∂L/∂β into the GradSlots buffers. Because Forward normalizes
// with running statistics that are constants with respect to the current
// sample, the gradient is exact and per-feature local:
//
//	dL/dγ_i += upstream_i · xHat_i
//	dL/dβ_i += upstream_i
//	dL/dx_i  = upstream_i · γ_i · invSd_i     (γ_i = 1 when affine disabled)
//
// Must be called after Forward on the same input (uses cached xHat, invSd).
//
// AI-Meta:
//   - Purpose: Backprop through BatchNorm; exact because running stats are sample-constants.
//   - Concurrency: NotSafe; reads cached Forward state.
//   - Related: [BatchNorm], [Forward], [GradSlots], [ApplyGradSGD].
//   - Stability: Stable.
func (b *BatchNorm[T]) Backward(upstream []T) []T {
	n := len(upstream)
	dx := make([]T, n)
	if n == 0 || len(b.xHat) < min(n, b.features) {
		return dx
	}
	for i := 0; i < n && i < b.features; i++ {
		u := upstream[i]
		if b.affine {
			b.gammaGrad[i] += u * b.xHat[i]
			b.betaGrad[i] += u
			u *= b.gamma[i]
		}
		dx[i] = u * b.invSd[i]
	}
	return dx
}

// ApplyGradSGD updates γ and β in-place (w -= lr·grad) and zeroes the
// gradient buffers, matching the inline-SGD convention shared by the conv
// prefix and the dense norm path. No-op when affine is disabled.
//
// AI-Meta:
//   - Purpose: Inline SGD update for BatchNorm affine params.
//   - Concurrency: NotSafe.
//   - Related: [BatchNorm], [Backward], [GradSlots].
//   - Stability: Stable.
func (b *BatchNorm[T]) ApplyGradSGD(lr T) {
	if !b.affine {
		return
	}
	for i := range b.gamma {
		b.gamma[i] -= lr * b.gammaGrad[i]
		b.gammaGrad[i] = 0
	}
	for i := range b.beta {
		b.beta[i] -= lr * b.betaGrad[i]
		b.betaGrad[i] = 0
	}
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
		Eps         T     `json:"eps"`
		Momentum    T     `json:"momentum"`
		RunningMean []T   `json:"running_mean"`
		RunningVar  []T   `json:"running_var"`
		Gamma       []T   `json:"gamma,omitempty"`
		Beta        []T   `json:"beta,omitempty"`
		Features    int   `json:"features"`
		Mode        int32 `json:"mode"`
		Affine      bool  `json:"affine"`
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
		Eps         T     `json:"eps"`
		Momentum    T     `json:"momentum"`
		RunningMean []T   `json:"running_mean"`
		RunningVar  []T   `json:"running_var"`
		Gamma       []T   `json:"gamma,omitempty"`
		Beta        []T   `json:"beta,omitempty"`
		Features    int   `json:"features"`
		Mode        int32 `json:"mode"`
		Affine      bool  `json:"affine"`
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
	// Payloads from the pre-v0.11 scalar-stat schema (or hand-written ones)
	// may lack per-feature slices — restore the identity defaults.
	if len(b.runningMean) != aux.Features {
		b.runningMean = make([]T, aux.Features)
	}
	if len(b.runningVar) != aux.Features {
		b.runningVar = make([]T, aux.Features)
		for i := range b.runningVar {
			b.runningVar[i] = 1
		}
	}
	b.xHat = make([]T, aux.Features)
	b.invSd = make([]T, aux.Features)
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
