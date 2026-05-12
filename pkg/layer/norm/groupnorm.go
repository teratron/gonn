package norm

import (
	"encoding/json"
	"sync/atomic"

	"github.com/teratron/gonn/pkg/utils"
)

// ErrGroupSizeMismatch is returned by NewGroupNorm when the feature count is not
// evenly divisible by the requested group count.
//
// AI-Meta:
//   - Purpose: Sentinel for GroupNorm construction failure when features % groups != 0.
//   - Usage: errors.Is(err, utils.ErrUserConfig) after NewGroupNorm returns an error.
//   - Related: [GroupNorm], [NewGroupNorm].
//   - Stability: Stable.
var ErrGroupSizeMismatch = utils.Newf(utils.ErrUserConfig,
	"GroupNorm: features must be divisible by groups")

// GroupNormOption is a functional option for NewGroupNorm.
//
// AI-Meta:
//   - Purpose: Functional option type for GroupNorm construction.
//   - Related: [NewGroupNorm], [WithGroupNormEps], [WithGroupNormAffine].
//   - Stability: Stable.
type GroupNormOption[T utils.Float] func(*GroupNorm[T])

// WithGroupNormEps sets the epsilon value for GroupNorm. Defaults to 1e-5.
//
// AI-Meta:
//   - Purpose: Override variance-floor epsilon for numerical stability in GroupNorm.
//   - Related: [GroupNormOption], [NewGroupNorm].
//   - Stability: Stable.
func WithGroupNormEps[T utils.Float](eps T) GroupNormOption[T] {
	return func(g *GroupNorm[T]) { g.eps = eps }
}

// WithGroupNormAffine controls whether learnable affine parameters are allocated.
// Defaults to true.
//
// AI-Meta:
//   - Purpose: Disable per-feature affine transform in GroupNorm.
//   - Related: [GroupNormOption], [NewGroupNorm].
//   - Stability: Stable.
func WithGroupNormAffine[T utils.Float](a bool) GroupNormOption[T] {
	return func(g *GroupNorm[T]) { g.affine = a }
}

// GroupNorm is a group normalization layer that partitions the feature vector into
// G equal groups and normalizes within each group independently (NORM-2). Like
// LayerNorm it has no running statistics. SetMode has no effect on the computation.
//
// AI-Meta:
//   - Purpose: Group-wise normalization that partitions features into G groups; suitable for any batch size.
//   - Usage: gn := norm.NewGroupNorm[float32](features, groups); out := gn.Forward(x).
//   - Concurrency: SetMode is goroutine-safe; Forward must be called from a single goroutine.
//   - Errors: NewGroupNorm returns ErrGroupSizeMismatch if features % groups != 0.
//   - Related: [NewGroupNorm], [Normalizer], [BatchNorm], [LayerNorm].
//   - Stability: Stable.
type GroupNorm[T utils.Float] struct {
	features  int
	groups    int
	eps       T
	affine    bool
	gamma     []T
	beta      []T
	gammaGrad []T
	betaGrad  []T
	mode      atomic.Int32
}

// NewGroupNorm constructs a GroupNorm layer. Returns ErrGroupSizeMismatch if
// features is not divisible by groups.
//
// AI-Meta:
//   - Purpose: Construct a GroupNorm with per-feature affine params and G-group partition.
//   - Usage: gn, err := norm.NewGroupNorm[float32](hiddenSize, 4).
//   - Errors: ErrGroupSizeMismatch if features % groups != 0 or features < 1 or groups < 1.
//   - Related: [GroupNorm], [GroupNormOption].
//   - Stability: Stable.
func NewGroupNorm[T utils.Float](features, groups int, opts ...GroupNormOption[T]) (*GroupNorm[T], error) {
	if features < 1 || groups < 1 {
		return nil, utils.Newf(utils.ErrUserConfig,
			"GroupNorm: features (%d) and groups (%d) must both be >= 1", features, groups)
	}
	if features%groups != 0 {
		return nil, utils.Newf(utils.ErrUserConfig,
			"GroupNorm: features (%d) must be divisible by groups (%d): %w", features, groups, ErrGroupSizeMismatch)
	}
	g := &GroupNorm[T]{
		features: features,
		groups:   groups,
		eps:      T(1e-5),
		affine:   true,
	}
	for _, opt := range opts {
		opt(g)
	}
	if g.affine {
		g.gamma = make([]T, features)
		g.beta = make([]T, features)
		g.gammaGrad = make([]T, features)
		g.betaGrad = make([]T, features)
		for i := range g.gamma {
			g.gamma[i] = 1
		}
	}
	return g, nil
}

// Forward partitions x into G groups and normalizes within each group independently.
// Returns a slice of the same length (NORM-1). SetMode has no effect.
//
// AI-Meta:
//   - Purpose: Normalize each of the G groups of the feature vector independently.
//   - Concurrency: NotSafe.
//   - Related: [GroupNorm], [Normalizer.Forward].
//   - Stability: Stable.
func (g *GroupNorm[T]) Forward(x []T) []T {
	if len(x) == 0 {
		return x
	}
	groupSize := g.features / g.groups
	xHat := make([]T, len(x))
	for grp := range g.groups {
		start := grp * groupSize
		end := start + groupSize
		slice := x[start:end]
		mean, variance := batchStats(slice)
		sd := stddev(variance, g.eps)
		for i, v := range slice {
			xHat[start+i] = (v - mean) / sd
		}
	}
	return applyAffine(xHat, g.gamma, g.beta)
}

// SetMode is accepted for Normalizer interface compatibility; GroupNorm's
// computation is always per-sample and is not affected by mode.
//
// AI-Meta:
//   - Purpose: Interface-compatible mode setter; GroupNorm is mode-agnostic.
//   - Concurrency: Safe.
//   - Related: [GroupNorm], [Normalizer.SetMode].
//   - Stability: Stable.
func (g *GroupNorm[T]) SetMode(m NormMode) {
	g.mode.Store(int32(m))
}

// GradSlots returns gradient accumulation slices for gamma and beta, or (nil, nil)
// when affine is disabled.
//
// AI-Meta:
//   - Purpose: Expose per-feature gradient slots for optimizer integration (NORM-7).
//   - Related: [GroupNorm], [Normalizer.GradSlots].
//   - Stability: Stable.
func (g *GroupNorm[T]) GradSlots() (gamma, beta []T) {
	if !g.affine {
		return nil, nil
	}
	return g.gammaGrad, g.betaGrad
}

// InputSize returns the expected input feature count.
//
// AI-Meta:
//   - Purpose: Report input dimensionality for topology integration.
//   - Related: [GroupNorm], [Normalizer.InputSize].
//   - Stability: Stable.
func (g *GroupNorm[T]) InputSize() int { return g.features }

// OutputSize always equals InputSize.
//
// AI-Meta:
//   - Purpose: Report output dimensionality; always equals InputSize for GroupNorm.
//   - Related: [GroupNorm], [Normalizer.OutputSize].
//   - Stability: Stable.
func (g *GroupNorm[T]) OutputSize() int { return g.features }

// MarshalJSON encodes GroupNorm state for persistence.
//
// AI-Meta:
//   - Purpose: JSON serialization for checkpoint and persistence round-trip.
//   - Related: [GroupNorm], [UnmarshalJSON].
//   - Stability: Stable.
func (g *GroupNorm[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Features int  `json:"features"`
		Groups   int  `json:"groups"`
		Eps      T    `json:"eps"`
		Affine   bool `json:"affine"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
	}{
		Features: g.features,
		Groups:   g.groups,
		Eps:      g.eps,
		Affine:   g.affine,
		Gamma:    g.gamma,
		Beta:     g.beta,
	})
}

// UnmarshalJSON restores GroupNorm state from a JSON payload.
//
// AI-Meta:
//   - Purpose: JSON deserialization for checkpoint and persistence round-trip.
//   - Related: [GroupNorm], [MarshalJSON].
//   - Stability: Stable.
func (g *GroupNorm[T]) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Features int  `json:"features"`
		Groups   int  `json:"groups"`
		Eps      T    `json:"eps"`
		Affine   bool `json:"affine"`
		Gamma    []T  `json:"gamma,omitempty"`
		Beta     []T  `json:"beta,omitempty"`
	}{}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	g.features = aux.Features
	g.groups = aux.Groups
	g.eps = aux.Eps
	g.affine = aux.Affine
	g.gamma = aux.Gamma
	g.beta = aux.Beta
	g.gammaGrad = make([]T, aux.Features)
	g.betaGrad = make([]T, aux.Features)
	return nil
}
