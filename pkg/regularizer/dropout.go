package regularizer

import (
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/utils"
)

// Dropout applies inverted Bernoulli dropout: each activation is retained with
// probability p and scaled by 1/p; dropped activations are set to zero.
//
// ApplyMask(acts, false) is a strict no-op — inference always passes through
// the full activation (REG-3). RNG state is never consumed on inference paths.
//
// RNG is seeded from the network's global seed when WithSeed is provided,
// otherwise from time.Now().UnixNano() via utils.NewRNG(0).
//
// AI-Meta:
//   - Purpose: Inverted Bernoulli dropout regularizer; reduces co-adaptation of hidden units.
//   - Usage: reg := regularizer.NewDropout[float32](0.8); passed to nn.WithRegularizer.
//   - Concurrency: SingleGoroutine; ApplyMask mutates the activation slice.
//   - Related: [Regularizer], [Compose], [NewL2].
//   - Stability: Stable.
type Dropout[T utils.Float] struct {
	rng  *rand.Rand
	mask []bool
	p    float64
}

// compile-time interface verification (C26).
var _ Regularizer[float32] = (*Dropout[float32])(nil)

// NewDropout returns a Dropout regularizer with retention probability p (0 < p ≤ 1).
// The RNG is seeded non-deterministically (utils.NewRNG(0)).
//
// AI-Meta:
//   - Purpose: Construct a Dropout regularizer with the given retention probability.
//   - Usage: reg := regularizer.NewDropout[float32](0.8).
//   - Related: [Dropout], [NewDropoutSeeded].
//   - Stability: Stable.
func NewDropout[T utils.Float](p float64) *Dropout[T] {
	rng, _ := utils.NewRNG(0)
	return &Dropout[T]{p: p, rng: rng}
}

// NewDropoutSeeded returns a Dropout regularizer with a deterministic RNG seed.
//
// AI-Meta:
//   - Purpose: Construct a deterministically seeded Dropout for reproducible training runs.
//   - Usage: reg := regularizer.NewDropoutSeeded[float32](0.8, 42).
//   - Related: [Dropout], [NewDropout].
//   - Stability: Stable.
func NewDropoutSeeded[T utils.Float](p float64, seed uint64) *Dropout[T] {
	rng, _ := utils.NewRNG(seed)
	return &Dropout[T]{p: p, rng: rng}
}

// Penalty returns zero — Dropout has no weight penalty term.
func (d *Dropout[T]) Penalty(_ []T) T { return 0 }

// WeightGrad returns zero — Dropout regularizes via activation masking, not
// weight decay.
func (d *Dropout[T]) WeightGrad(_ T) T { return 0 }

// ApplyMask applies the inverted Bernoulli mask when training=true.
// Each element is zeroed with probability (1-p); retained elements are
// scaled by 1/p (REG-5). Returns acts unchanged when training=false (REG-3).
// Stores the retain mask for use by BackwardMask on the same pass.
func (d *Dropout[T]) ApplyMask(acts []T, training bool) []T {
	if !training {
		d.mask = d.mask[:0]
		return acts
	}
	if cap(d.mask) < len(acts) {
		d.mask = make([]bool, len(acts))
	} else {
		d.mask = d.mask[:len(acts)]
	}
	scale := T(1.0 / d.p)
	for i, a := range acts {
		if d.rng.Float64() < d.p {
			acts[i] = a * scale
			d.mask[i] = true
		} else {
			acts[i] = 0
			d.mask[i] = false
		}
	}
	return acts
}

// BackwardMask applies the retain mask from the last ApplyMask call to upstream,
// returning ∂L/∂acts before the dropout gate. Dropped positions get zero gradient;
// retained positions are scaled by 1/p to match the forward scaling.
// Returns upstream unchanged when training was false (empty mask).
func (d *Dropout[T]) BackwardMask(upstream []T) []T {
	if len(d.mask) == 0 {
		return upstream
	}
	scale := T(1.0 / d.p)
	out := make([]T, len(upstream))
	for i, u := range upstream {
		if i < len(d.mask) && d.mask[i] {
			out[i] = u * scale
		}
	}
	return out
}
