// Package utils — RNG plumbing and weight-init sampling helpers.
//
// This file implements [l2-init-impl] §5.1 (helpers) and §5.2 (RNG seed
// contract). One *rand.Rand is owned by each *NN[T] instance and re-used
// for every sampling call (see WI-1, WI-2 in l1-weight-initialization).
//
// The deterministic source is [math/rand/v2.PCG], chosen because:
//   - reproducible across architectures (no float-rounding source like Mersenne Twister state),
//   - exposes MarshalBinary/UnmarshalBinary required by checkpoint persistence,
//   - faster than the legacy Mersenne Twister and stdlib-only (C29).
package utils

import (
	"math"
	"math/rand/v2"
	"time"
)

// NewRNG returns a deterministic random source seeded by seed. When seed
// is 0 the function falls back to time.Now().UnixNano() — the resulting
// session is non-reproducible by design (per WI-2: zero seed signals
// "I do not care about reproducibility"). Callers that persist the seed
// should record the value returned by the second result.
//
// The returned *rand.Rand is NOT safe for concurrent use. The training
// loop is single-goroutine; concurrent callers must wrap their own mutex.
//
// AI-Meta:
//   - Purpose: Create a seeded *rand.Rand for weight initialization; returns the effective seed.
//   - Usage: rng, seed := utils.NewRNG(0) — seed=0 uses wall clock for non-reproducible training.
//   - Concurrency: NotSafe; the returned *rand.Rand must not be shared across goroutines.
//   - Related: [XavierUniform], [HeNormal], [Uniform].
func NewRNG(seed uint64) (*rand.Rand, uint64) {
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	// Two different stream values keep PCG state space distinct between
	// network instances that share the wall-clock seed (sub-millisecond
	// double-Compile is theoretically possible).
	return rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15)), seed
}

// XavierUniform samples one weight from the Glorot uniform distribution
// U[-a, a] where a = sqrt(6 / (fanIn + fanOut)).
//
// Reference: Glorot, X. and Bengio, Y. (2010). Understanding the difficulty
// of training deep feedforward neural networks.
//
// XavierUniform panics when rng is nil — a nil source is always a
// programming bug; recoverable misconfiguration belongs in Compile().
//
// AI-Meta:
//   - Purpose: Sample one weight from the Glorot uniform distribution, suited for sigmoid/tanh layers.
//   - Usage: w := utils.XavierUniform[float32](rng, fanIn, fanOut).
//   - Concurrency: NotSafe; rng must not be shared across goroutines.
//   - Related: [NewRNG], [HeNormal], [Uniform].
func XavierUniform[T Float](rng *rand.Rand, fanIn, fanOut int) T {
	if rng == nil {
		panic("utils.XavierUniform: nil *rand.Rand")
	}
	if fanIn+fanOut <= 0 {
		// Degenerate topology — fall back to standard Uniform to avoid
		// division-by-zero. Callers should validate fan counts upstream
		// (T-1C02 layer constructors enforce this).
		return Uniform[T](rng)
	}
	a := math.Sqrt(6.0 / float64(fanIn+fanOut))
	return T(rng.Float64()*2*a - a)
}

// HeNormal samples one weight from the He normal distribution
// N(0, sigma^2) where sigma = sqrt(2 / fanIn).
//
// Reference: He, K. et al. (2015). Delving deep into rectifiers.
//
// HeNormal panics when rng is nil.
//
// AI-Meta:
//   - Purpose: Sample one weight from the He normal distribution, suited for ReLU-family layers.
//   - Usage: w := utils.HeNormal[float32](rng, fanIn).
//   - Concurrency: NotSafe; rng must not be shared across goroutines.
//   - Related: [NewRNG], [XavierUniform], [Uniform].
func HeNormal[T Float](rng *rand.Rand, fanIn int) T {
	if rng == nil {
		panic("utils.HeNormal: nil *rand.Rand")
	}
	if fanIn <= 0 {
		return Uniform[T](rng)
	}
	sigma := math.Sqrt(2.0 / float64(fanIn))
	return T(rng.NormFloat64() * sigma)
}

// Uniform samples one weight from U[-1, 1).
//
// Uniform panics when rng is nil.
//
// AI-Meta:
//   - Purpose: Sample one weight from U[-1, 1) — fallback initializer when fan counts are unavailable.
//   - Usage: w := utils.Uniform[float32](rng).
//   - Concurrency: NotSafe; rng must not be shared across goroutines.
//   - Related: [NewRNG], [XavierUniform], [HeNormal].
func Uniform[T Float](rng *rand.Rand) T {
	if rng == nil {
		panic("utils.Uniform: nil *rand.Rand")
	}
	return T(rng.Float64()*2 - 1)
}
