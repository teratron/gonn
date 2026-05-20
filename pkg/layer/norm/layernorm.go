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
	eps       T
	gamma     []T
	beta      []T
	gammaGrad []T
	betaGrad  []T
	// xHat and invSd are cached by Forward for use in Backward.
	xHat  []T
	invSd T
	// xHatBuf and invSdBuf store per-position caches for ForwardSeq/BackwardSeq.
	xHatBuf  []T
	invSdBuf []T
	features int
	mode     atomic.Int32
	affine   bool
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
// Caches the normalized vector and inverse std-dev for use by Backward.
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
	if cap(l.xHat) < len(x) {
		l.xHat = make([]T, len(x))
	} else {
		l.xHat = l.xHat[:len(x)]
	}
	for i, v := range x {
		l.xHat[i] = (v - mean) / sd
	}
	l.invSd = 1 / sd
	return applyAffine(l.xHat, l.gamma, l.beta)
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

// Backward computes the input gradient ∂L/∂x given the upstream gradient ∂L/∂y,
// accumulating ∂L/∂γ and ∂L/∂β into the GradSlots buffers.
// Must be called after Forward on the same input (uses cached xHat and invSd).
//
// Gradient derivation (all indices over N = len(upstream)):
//
//	dL/dxHat_i = upstream_i * γ_i     (or upstream_i when affine disabled)
//	dL/dx_i    = invSd/N * (N·dL/dxHat_i − Σ dL/dxHat_j − xHat_i·Σ(dL/dxHat_j·xHat_j))
//
// AI-Meta:
//   - Purpose: Backprop through LayerNorm; accumulates affine gradients, returns ∂L/∂x.
//   - Concurrency: NotSafe; reads cached Forward state (xHat, invSd).
//   - Related: [LayerNorm], [Forward], [GradSlots], [ApplyGradSGD].
//   - Stability: Stable.
func (l *LayerNorm[T]) Backward(upstream []T) []T {
	n := len(upstream)
	if n == 0 || len(l.xHat) == 0 {
		return make([]T, n)
	}

	dxHat := make([]T, n)
	var sumDxHat, sumDxHatXHat T
	for i, u := range upstream {
		g := u
		if l.affine {
			if l.gammaGrad != nil {
				l.gammaGrad[i] += u * l.xHat[i]
			}
			if l.betaGrad != nil {
				l.betaGrad[i] += u
			}
			g *= l.gamma[i]
		}
		dxHat[i] = g
		sumDxHat += g
		sumDxHatXHat += g * l.xHat[i]
	}

	dx := make([]T, n)
	scale := l.invSd / T(n)
	for i, d := range dxHat {
		dx[i] = scale * (T(n)*d - sumDxHat - l.xHat[i]*sumDxHatXHat)
	}
	return dx
}

// ApplyGradSGD updates the affine parameters gamma and beta in-place using the
// accumulated gradient buffers: w -= lr * grad. Zeros the gradient buffers after
// the update, matching the conv-prefix inline-SGD convention (applyConvSGD).
//
// AI-Meta:
//   - Purpose: Inline SGD update for LayerNorm affine params in the conv-prefix backward path.
//   - Concurrency: NotSafe.
//   - Related: [LayerNorm], [Backward], [GradSlots].
//   - Stability: Stable.
func (l *LayerNorm[T]) ApplyGradSGD(lr T) {
	if !l.affine {
		return
	}
	for i := range l.gamma {
		l.gamma[i] -= lr * l.gammaGrad[i]
		l.gammaGrad[i] = 0
	}
	for i := range l.beta {
		l.beta[i] -= lr * l.betaGrad[i]
		l.betaGrad[i] = 0
	}
}

// ForwardSeq applies LayerNorm to each of seqLen positions in a flat
// [seqLen*features] input tensor and returns a flat output of the same shape.
// Caches per-position xHat and invSd in xHatBuf/invSdBuf for BackwardSeq.
func (l *LayerNorm[T]) ForwardSeq(x []T, seqLen int) []T {
	needed := seqLen * l.features
	if len(l.xHatBuf) < needed {
		l.xHatBuf = make([]T, needed)
	}
	if len(l.invSdBuf) < seqLen {
		l.invSdBuf = make([]T, seqLen)
	}
	out := make([]T, needed)
	for p := range seqLen {
		base := p * l.features
		posOut := l.Forward(x[base : base+l.features])
		copy(out[base:base+l.features], posOut)
		copy(l.xHatBuf[base:base+l.features], l.xHat)
		l.invSdBuf[p] = l.invSd
	}
	return out
}

// BackwardSeq reverses ForwardSeq for all seqLen positions. Accumulates
// affine parameter gradients from every position into the shared gamma/beta
// grad buffers and returns ∂L/∂x with shape [seqLen*features].
// Must be called after ForwardSeq on the same input.
func (l *LayerNorm[T]) BackwardSeq(upstream []T, seqLen int) []T {
	dx := make([]T, seqLen*l.features)
	for p := range seqLen {
		base := p * l.features
		l.xHat = l.xHatBuf[base : base+l.features]
		l.invSd = l.invSdBuf[p]
		posGrad := l.Backward(upstream[base : base+l.features])
		copy(dx[base:base+l.features], posGrad)
	}
	return dx
}

// Params returns the live gamma and beta parameter slices.
// Modifying the returned slices directly modifies the LayerNorm parameters.
// Returns (nil, nil) when affine is disabled.
func (l *LayerNorm[T]) Params() (gamma, beta []T) {
	if !l.affine {
		return nil, nil
	}
	return l.gamma, l.beta
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
		Eps      T    `json:"eps"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
		Features int  `json:"features"`
		Affine   bool `json:"affine"`
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
		Eps      T    `json:"eps"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
		Features int  `json:"features"`
		Affine   bool `json:"affine"`
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
