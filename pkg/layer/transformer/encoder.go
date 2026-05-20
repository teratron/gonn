package transformer

import (
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// EncoderBlock implements a single Transformer encoder layer:
//
//	Post-norm:  y = LN₂(Z + FFN(LN₁(X + Attn(X))))
//	Pre-norm:   y = X + Attn(LN₁(X)); Z = X + FFN(LN₂(Z))  (TRANS-3)
//
// Dropout is applied at the three TRANS-C7 positions when training=true.
// Children are owned as named typed fields (TRANS §3 typed ownership) to
// allow zero-cost hot-path access without interface-dispatch overhead.
//
// AI-Meta:
//   - Purpose: Single Transformer encoder layer with residuals, two LayerNorms, and position-wise FFN.
//   - Usage: blk := transformer.NewEncoderBlock[float32](cfg); blk.Init(rng).
//   - Concurrency: NotSafe; Forward/Backward mutate per-call cache buffers.
//   - Related: [DecoderBlock], [Stack], [TransformerConfig], [layer.Layer].
//   - Stability: Experimental.
type EncoderBlock[T utils.Float] struct {
	Cfg   TransformerConfig[T]
	Attn  *attention.MultiHeadAttention[T]
	Norm1 *norm.LayerNorm[T]
	Norm2 *norm.LayerNorm[T]
	FFN   *ffn[T]
	Drop1 *regularizer.Dropout[T] // post-attention dropout (TRANS-C7 position 1)
	Drop2 *regularizer.Dropout[T] // post-FFN dropout (TRANS-C7 position 2)
	// Forward-cache buffers reused across calls (PERF-4).
	bufAttn []T // Attn output (SeqLen*Dmodel)
	bufZ    []T // post-first-residual (SeqLen*Dmodel)
	bufF    []T // FFN output (SeqLen*Dmodel)
	// backward-cache
	cacheX  []T // input to Forward
	cacheZ1 []T // post-first-LN (pre-FFN input)
	// training mode — controls Dropout.ApplyMask (false = inference, no dropout)
	training bool
}

// NewEncoderBlock constructs an EncoderBlock for the given config.
// Applies default substitutions: Activation → ReLU when zero (TRANS-C4),
// Dff → 4·Dmodel when zero (TRANS-C5).
// Call Init(rng) before use to Xavier-initialize all weight matrices.
//
// AI-Meta:
//   - Purpose: Allocate an EncoderBlock with default-filled config; weights zeroed until Init.
//   - Usage: blk := transformer.NewEncoderBlock[float64](cfg); blk.Init(rng).
//   - Related: [EncoderBlock.Init], [TransformerConfig].
//   - Stability: Experimental.
func NewEncoderBlock[T utils.Float](cfg TransformerConfig[T]) *EncoderBlock[T] {
	if cfg.Activation == 0 {
		cfg.Activation = defaultActivation
	}
	if cfg.Dff == 0 {
		cfg.Dff = 4 * cfg.Dmodel
	}
	p := retentionProb(cfg.DropoutRate)
	return &EncoderBlock[T]{
		Cfg:   cfg,
		Attn:  attention.NewMultiHeadAttention[T](cfg.SeqLen, cfg.Dmodel, cfg.NumHeads, false),
		Norm1: norm.NewLayerNorm[T](cfg.Dmodel),
		Norm2: norm.NewLayerNorm[T](cfg.Dmodel),
		FFN:   newFFN[T](cfg.SeqLen, cfg.Dmodel, cfg.Dff, cfg.Activation),
		Drop1: regularizer.NewDropout[T](p),
		Drop2: regularizer.NewDropout[T](p),
	}
}

// defaultActivation is the substituted activation type when the config field
// is zero (zero-value of activation.Type is ELISH, not ReLU — TRANS-C4).
var defaultActivation = activation.ReLU

// retentionProb converts a dropout rate (fraction dropped) to the retention
// probability expected by regularizer.NewDropout. A zero rate → 1.0 (keep all).
func retentionProb[T utils.Float](rate T) float64 {
	if rate <= 0 {
		return 1.0
	}
	if rate >= 1 {
		return 0.01 // guard against misconfiguration
	}
	return float64(1 - rate)
}

// Init Xavier-initializes all weight matrices and pre-allocates forward/backward
// cache buffers. Each child receives an independent RNG stream forked from rng
// (TRANS-C8 seed isolation).
//
// AI-Meta:
//   - Purpose: Initialize weights and pre-allocate caches; must be called before Forward.
//   - Concurrency: NotSafe.
//   - Related: [EncoderBlock], [NewEncoderBlock].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) Init(rng *rand.Rand) {
	s0, s1 := rng.Uint64(), rng.Uint64()
	e.Attn.Init(rand.New(rand.NewPCG(s0, s0^0x9E3779B9)))
	e.FFN.Init(rand.New(rand.NewPCG(s1, s1^0x9E3779B9)))
	// Pre-allocate cache buffers once (PERF-4).
	size := e.Cfg.SeqLen * e.Cfg.Dmodel
	e.bufAttn = make([]T, size)
	e.bufZ = make([]T, size)
	e.bufF = make([]T, size)
	e.cacheX = make([]T, size)
	e.cacheZ1 = make([]T, size)
}

// SetTraining controls whether Dropout is applied during Forward/Backward.
// Set to true during training passes and false for inference.
func (e *EncoderBlock[T]) SetTraining(training bool) {
	e.training = training
}

// Forward applies the post-norm (default) or pre-norm (when Cfg.PreNorm=true)
// encoder sub-layers and returns the output tensor of shape SeqLen*Dmodel.
//
// Post-norm (TRANS-2):
//
//	z1  = Drop1(Attn(x)) ; add(x, z1) ; z1 = Norm1(x+z1)
//	out = Drop2(FFN(z1)) ; add(z1, out) ; out = Norm2(z1+out)
//
// AI-Meta:
//   - Purpose: Encoder forward pass; preserves shape SeqLen*Dmodel (TRANS-1).
//   - Concurrency: NotSafe; mutates cache buffers.
//   - Related: [EncoderBlock.Backward], [EncoderBlock.Init].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) Forward(x []T) []T {
	if e.Cfg.PreNorm {
		return e.forwardPreNorm(x)
	}
	return e.forwardPostNorm(x)
}

// forwardPostNorm: LN after residual add (standard BERT-style).
func (e *EncoderBlock[T]) forwardPostNorm(x []T) []T {
	seqLen, dmodel := e.Cfg.SeqLen, e.Cfg.Dmodel

	// --- sub-layer 1: self-attention ---
	attnOut := e.Attn.Forward(x)
	attnOut = e.Drop1.ApplyMask(attnOut, e.training)
	// residual 1: bufZ = x + attnOut
	copy(e.bufZ, x)
	addInPlace(e.bufZ, attnOut)
	// LN1 applied per position; result stored in cacheZ1 (= FFN input).
	for p := range seqLen {
		base := p * dmodel
		pos := e.Norm1.Forward(e.bufZ[base : base+dmodel])
		copy(e.cacheZ1[base:base+dmodel], pos)
	}

	// --- sub-layer 2: FFN ---
	ffnOut := e.FFN.Forward(e.cacheZ1)
	ffnOut = e.Drop2.ApplyMask(ffnOut, e.training)
	// residual 2: bufF = cacheZ1 + ffnOut
	copy(e.bufF, e.cacheZ1)
	addInPlace(e.bufF, ffnOut)
	// LN2 applied per position.
	out := make([]T, seqLen*dmodel)
	for p := range seqLen {
		base := p * dmodel
		pos := e.Norm2.Forward(e.bufF[base : base+dmodel])
		copy(out[base:base+dmodel], pos)
	}

	copy(e.cacheX, x)
	return out
}

// forwardPreNorm: LN before sub-layer (recommended for deep stacks, TRANS-3).
// Implemented in T-18A07; placeholder routes to post-norm until then.
func (e *EncoderBlock[T]) forwardPreNorm(x []T) []T {
	// T-18A07 fills this in. For now identical to post-norm to keep the encoder
	// buildable; the pre-norm branch is gated by T-18A07 tests.
	return e.forwardPostNorm(x)
}

// Backward computes ∂L/∂x given the upstream gradient and accumulates
// parameter gradients in all children's GradSlots buffers.
// Must be called after Forward on the same input.
//
// Post-norm reverse cascade (TRANS-8):
//
//	upstream → Norm2 → residual split → FFN → Drop2 → Norm1 → residual split → Attn → Drop1 → ∂L/∂x
//
// AI-Meta:
//   - Purpose: Encoder backward pass; returns ∂L/∂x.
//   - Concurrency: NotSafe.
//   - Related: [EncoderBlock.Forward], [EncoderBlock.ApplyGradSGD].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) Backward(upstream []T) []T {
	if e.Cfg.PreNorm {
		return e.backwardPreNorm(upstream)
	}
	return e.backwardPostNorm(upstream)
}

// backwardPostNorm reverses the post-norm forward chain.
// NOTE: Norm1/Norm2 caches hold only the last-forwarded position; full
// per-position cache support is completed in T-18A05.
func (e *EncoderBlock[T]) backwardPostNorm(upstream []T) []T {
	seqLen, dmodel := e.Cfg.SeqLen, e.Cfg.Dmodel

	// reverse LN2 per position
	dBufF := make([]T, seqLen*dmodel)
	for p := range seqLen {
		base := p * dmodel
		pos := e.Norm2.Backward(upstream[base : base+dmodel])
		copy(dBufF[base:base+dmodel], pos)
	}

	// reverse residual 2: gradient flows to both FFN path and z1 path
	dFFNOut := make([]T, len(dBufF))
	copy(dFFNOut, dBufF)
	dZ1fromRes2 := dBufF

	// reverse Drop2 (mask-backward wired in T-18A08; pass-through for now)
	// reverse FFN (full sequence)
	dZ1fromFFN := e.FFN.Backward(dFFNOut)

	// combine z1 gradients
	dZ1 := make([]T, seqLen*dmodel)
	copy(dZ1, dZ1fromRes2)
	addInPlace(dZ1, dZ1fromFFN)

	// reverse LN1 per position
	dBufZ := make([]T, seqLen*dmodel)
	for p := range seqLen {
		base := p * dmodel
		pos := e.Norm1.Backward(dZ1[base : base+dmodel])
		copy(dBufZ[base:base+dmodel], pos)
	}

	// reverse residual 1: gradient flows to both Attn path and input path
	dAttnOut := make([]T, len(dBufZ))
	copy(dAttnOut, dBufZ)
	dXfromRes1 := dBufZ

	// reverse Drop1 (mask-backward wired in T-18A08; pass-through for now)
	// reverse Attn (full sequence)
	dXfromAttn := e.Attn.Backward(dAttnOut)

	// combine input gradients
	dx := make([]T, seqLen*dmodel)
	copy(dx, dXfromRes1)
	addInPlace(dx, dXfromAttn)

	return dx
}

// backwardPreNorm reverses the pre-norm forward chain (filled in T-18A07).
func (e *EncoderBlock[T]) backwardPreNorm(upstream []T) []T {
	return e.backwardPostNorm(upstream)
}

// GradSlots satisfies layer.Layer[T]. Returns (nil, nil) — the EncoderBlock
// updates parameters exclusively via ApplyGradSGD (inline-SGD conv-prefix path).
func (e *EncoderBlock[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// InputSize returns SeqLen * Dmodel (TRANS-1).
func (e *EncoderBlock[T]) InputSize() int { return e.Cfg.SeqLen * e.Cfg.Dmodel }

// OutputSize returns SeqLen * Dmodel (TRANS-1).
func (e *EncoderBlock[T]) OutputSize() int { return e.Cfg.SeqLen * e.Cfg.Dmodel }

// ApplyGradSGD fans the inline SGD step to every child with learnable parameters
// and zeroes all gradient buffers.
//
// AI-Meta:
//   - Purpose: Drive inline SGD update across all child parameter groups.
//   - Related: [EncoderBlock.Backward], [attention.MultiHeadAttention.ApplyGradSGD].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) ApplyGradSGD(lr T) {
	e.Attn.ApplyGradSGD(lr)
	e.Norm1.ApplyGradSGD(lr)
	e.Norm2.ApplyGradSGD(lr)
	e.FFN.ApplyGradSGD(lr)
}

// SetPaddingMask forwards the padding mask to the inner MHA (TRANS-10).
// LayerNorm and FFN are mask-agnostic; only attention uses the mask.
//
// AI-Meta:
//   - Purpose: Forward the padding mask to the inner MultiHeadAttention layer.
//   - Related: [attention.MaskedLayer], [attention.MultiHeadAttention.SetPaddingMask].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) SetPaddingMask(mask []bool) {
	e.Attn.SetPaddingMask(mask)
}

// Compile-time interface assertions.
var (
	_ layer.Layer[float32]           = (*EncoderBlock[float32])(nil)
	_ layer.Layer[float64]           = (*EncoderBlock[float64])(nil)
	_ attention.MaskedLayer[float32] = (*EncoderBlock[float32])(nil)
	_ attention.MaskedLayer[float64] = (*EncoderBlock[float64])(nil)
	_ Block[float32]                 = (*EncoderBlock[float32])(nil)
	_ Block[float64]                 = (*EncoderBlock[float64])(nil)
)
