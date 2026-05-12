package norm

import (
	"encoding/json"
	"sync/atomic"

	"github.com/teratron/gonn/pkg/utils"
)

// LayerNormOption is a functional option for NewLayerNorm.
//
// AI-Meta:
//   - Purpose: Functional option type for LayerNorm construction.
//   - Usage: NewLayerNorm[float32](4, WithLayerNormEps[float32](1e-3)).
//   - Related: [NewLayerNorm], [WithLayerNormEps], [WithLayerNormAffine].
//   - Stability: Stable.
type LayerNormOption[T utils.Float] func(*LayerNorm[T])

// WithLayerNormEps sets the epsilon value for LayerNorm. Defaults to 1e-5.
//
// AI-Meta:
//   - Purpose: Override the variance-floor epsilon for numerical stability in LayerNorm.
//   - Related: [LayerNormOption], [NewLayerNorm].
//   - Stability: Stable.
func WithLayerNormEps[T utils.Float](eps T) LayerNormOption[T] {
	return func(l *LayerNorm[T]) { l.eps = eps }
}

// WithLayerNormAffine controls whether learnable affine parameters are allocated.
// When false, GradSlots returns (nil, nil). Defaults to true.
//
// AI-Meta:
//   - Purpose: Disable per-feature scale/shift to use LayerNorm as a pure normalizer.
//   - Related: [LayerNormOption], [NewLayerNorm].
//   - Stability: Stable.
func WithLayerNormAffine[T utils.Float](a bool) LayerNormOption[T] {
	return func(l *LayerNorm[T]) { l.affine = a }
}

// LayerNorm is a layer normalization layer that computes per-sample mean and
// variance across all features of the input vector. Unlike BatchNorm it has no
// running statistics — Forward is stateless with respect to the sample sequence.
// SetMode has no observable effect on LayerNorm's computation (forward is always
// per-sample), but the call is accepted for interface compatibility.
//
// AI-Meta:
//   - Purpose: Per-sample normalization across all features; no running statistics, suitable for any batch size.
//   - Usage: ln := norm.NewLayerNorm[float32](features); out := ln.Forward(x).
//   - Concurrency: SetMode is goroutine-safe; Forward must be called from a single goroutine.
//   - Related: [NewLayerNorm], [Normalizer], [BatchNorm], [GroupNorm].
//   - Stability: Stable.
type LayerNorm[T utils.Float] struct {
	features  int
	eps       T
	affine    bool
	gamma     []T
	beta      []T
	gammaGrad []T
	betaGrad  []T
	mode      atomic.Int32 // stored but not used — kept for interface parity
}

// NewLayerNorm constructs a LayerNorm for the given feature count.
// Default: eps=1e-5, affine=true with gamma=1 and beta=0.
//
// AI-Meta:
//   - Purpose: Construct a LayerNorm with optional per-feature affine transform.
//   - Usage: ln := norm.NewLayerNorm[float32](hiddenSize).
//   - Errors: Panics if features < 1.
//   - Related: [LayerNorm], [LayerNormOption].
//   - Stability: Stable.
func NewLayerNorm[T utils.Float](features int, opts ...LayerNormOption[T]) *LayerNorm[T] {
	if features < 1 {
		panic("norm.NewLayerNorm: features must be >= 1")
	}
	l := &LayerNorm[T]{
		features: features,
		eps:      T(1e-5),
		affine:   true,
	}
	for _, opt := range opts {
		opt(l)
	}
	if l.affine {
		l.gamma = make([]T, features)
		l.beta = make([]T, features)
		l.gammaGrad = make([]T, features)
		l.betaGrad = make([]T, features)
		for i := range l.gamma {
			l.gamma[i] = 1
		}
	}
	return l
}

// Forward normalizes x per-sample (NORM-2: axis over x directly) and applies
// the optional affine transform. Returns a slice of the same length as x (NORM-1).
//
// AI-Meta:
//   - Purpose: Compute per-sample mean/variance normalization across all features.
//   - Concurrency: NotSafe.
//   - Related: [LayerNorm], [Normalizer.Forward].
//   - Stability: Stable.
func (l *LayerNorm[T]) Forward(x []T) []T {
	if len(x) == 0 {
		return x
	}
	mean, variance := batchStats(x)
	sd := stddev(variance, l.eps)
	xHat := make([]T, len(x))
	for i, v := range x {
		xHat[i] = (v - mean) / sd
	}
	return applyAffine(xHat, l.gamma, l.beta)
}

// SetMode is accepted for Normalizer interface compatibility; LayerNorm's
// computation is always per-sample regardless of mode.
//
// AI-Meta:
//   - Purpose: Interface-compatible mode setter; LayerNorm is mode-agnostic.
//   - Concurrency: Safe.
//   - Related: [LayerNorm], [Normalizer.SetMode].
//   - Stability: Stable.
func (l *LayerNorm[T]) SetMode(m NormMode) {
	l.mode.Store(int32(m))
}

// GradSlots returns gradient accumulation slices for gamma and beta, or (nil, nil)
// when affine is disabled.
//
// AI-Meta:
//   - Purpose: Expose per-feature gradient slots for optimizer integration (NORM-7).
//   - Related: [LayerNorm], [Normalizer.GradSlots].
//   - Stability: Stable.
func (l *LayerNorm[T]) GradSlots() (gamma, beta []T) {
	if !l.affine {
		return nil, nil
	}
	return l.gammaGrad, l.betaGrad
}

// InputSize returns the expected input feature count.
//
// AI-Meta:
//   - Purpose: Report input dimensionality for topology integration.
//   - Related: [LayerNorm], [Normalizer.InputSize].
//   - Stability: Stable.
func (l *LayerNorm[T]) InputSize() int { return l.features }

// OutputSize always equals InputSize.
//
// AI-Meta:
//   - Purpose: Report output dimensionality; always equals InputSize for LayerNorm.
//   - Related: [LayerNorm], [Normalizer.OutputSize].
//   - Stability: Stable.
func (l *LayerNorm[T]) OutputSize() int { return l.features }

// MarshalJSON encodes LayerNorm state for persistence.
//
// AI-Meta:
//   - Purpose: JSON serialization for checkpoint and persistence round-trip.
//   - Related: [LayerNorm], [UnmarshalJSON].
//   - Stability: Stable.
func (l *LayerNorm[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Features int  `json:"features"`
		Eps      T    `json:"eps"`
		Affine   bool `json:"affine"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
	}{
		Features: l.features,
		Eps:      l.eps,
		Affine:   l.affine,
		Gamma:    l.gamma,
		Beta:     l.beta,
	})
}

// UnmarshalJSON restores LayerNorm state from a JSON payload.
//
// AI-Meta:
//   - Purpose: JSON deserialization for checkpoint and persistence round-trip.
//   - Related: [LayerNorm], [MarshalJSON].
//   - Stability: Stable.
func (l *LayerNorm[T]) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Features int  `json:"features"`
		Eps      T    `json:"eps"`
		Affine   bool `json:"affine"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
	}{}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	l.features = aux.Features
	l.eps = aux.Eps
	l.affine = aux.Affine
	l.gamma = aux.Gamma
	l.beta = aux.Beta
	l.gammaGrad = make([]T, aux.Features)
	l.betaGrad = make([]T, aux.Features)
	return nil
}
