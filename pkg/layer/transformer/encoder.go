package transformer

import (
	"encoding/json"
	"fmt"
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
//	Pre-norm:   Y = X + Attn(LN₁(X)); Z = Y + FFN(LN₂(Y))  (TRANS-3)
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
	Drop2    *regularizer.Dropout[T]
	Attn     *attention.MultiHeadAttention[T]
	Norm1    *norm.LayerNorm[T]
	Norm2    *norm.LayerNorm[T]
	FFN      *ffn[T]
	Drop1    *regularizer.Dropout[T]
	bufAttn  []T
	bufZ     []T
	bufF     []T
	cacheX   []T
	cacheZ1  []T
	Cfg      TransformerConfig[T]
	training bool
}

// NewEncoderBlock constructs an EncoderBlock for the given config.
// Applies default substitutions: Activation → ReLU when zero (TRANS-C4),
// Dff → 4·Dmodel when zero (TRANS-C5).
// Call Init(rng) before use to Xavier-initialize all weight matrices.
//
// PreNorm note: pre-norm wiring (cfg.PreNorm=true) is recommended for stacks of
// 6+ layers because the gradient flows through the residual connections without
// passing through LayerNorm, which keeps the effective learning rate stable at
// depth. Post-norm is the default (BERT-style) and works well for shallower stacks.
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
	// Lazy-allocate cache buffers if Init has not been called yet (e.g. during
	// compile-time shape inference via computeConvChainOutput).
	if e.bufZ == nil {
		size := e.Cfg.SeqLen * e.Cfg.Dmodel
		e.bufAttn = make([]T, size)
		e.bufZ = make([]T, size)
		e.bufF = make([]T, size)
		e.cacheX = make([]T, size)
		e.cacheZ1 = make([]T, size)
	}
	if e.Cfg.PreNorm {
		return e.forwardPreNorm(x)
	}
	return e.forwardPostNorm(x)
}

// forwardPostNorm: LN after residual add (standard BERT-style).
func (e *EncoderBlock[T]) forwardPostNorm(x []T) []T {
	seqLen := e.Cfg.SeqLen

	// --- sub-layer 1: self-attention ---
	attnOut := e.Attn.Forward(x)
	attnOut = e.Drop1.ApplyMask(attnOut, e.training)
	// residual 1: bufZ = x + attnOut
	copy(e.bufZ, x)
	addInPlace(e.bufZ, attnOut)
	// LN1 per position via ForwardSeq; per-position cache stored for backward.
	copy(e.cacheZ1, e.Norm1.ForwardSeq(e.bufZ, seqLen))

	// --- sub-layer 2: FFN ---
	ffnOut := e.FFN.Forward(e.cacheZ1)
	ffnOut = e.Drop2.ApplyMask(ffnOut, e.training)
	// residual 2: bufF = cacheZ1 + ffnOut
	copy(e.bufF, e.cacheZ1)
	addInPlace(e.bufF, ffnOut)
	// LN2 per position via ForwardSeq.
	out := e.Norm2.ForwardSeq(e.bufF, seqLen)

	copy(e.cacheX, x)
	return out
}

// forwardPreNorm: LN before each sub-layer (TRANS-3).
//
//	Y = X + Drop1(Attn(LN₁(X)))
//	Z = Y + Drop2(FFN(LN₂(Y)))
//
// ForwardSeq caches per-position xHat in Norm1/Norm2 xHatBuf for BackwardSeq.
func (e *EncoderBlock[T]) forwardPreNorm(x []T) []T {
	seqLen := e.Cfg.SeqLen

	// sub-layer 1: attention on LN1-normalized input, then residual
	ln1Out := e.Norm1.ForwardSeq(x, seqLen)
	attnOut := e.Attn.Forward(ln1Out)
	attnOut = e.Drop1.ApplyMask(attnOut, e.training)
	copy(e.bufZ, x)
	addInPlace(e.bufZ, attnOut) // y = x + Drop1(Attn(LN1(x)))
	copy(e.cacheZ1, e.bufZ)     // cache y so LN2 sees it in ForwardSeq

	// sub-layer 2: FFN on LN2-normalized residual, then residual
	ln2Out := e.Norm2.ForwardSeq(e.cacheZ1, seqLen)
	ffnOut := e.FFN.Forward(ln2Out)
	ffnOut = e.Drop2.ApplyMask(ffnOut, e.training)
	copy(e.bufF, e.cacheZ1)
	addInPlace(e.bufF, ffnOut) // z = y + Drop2(FFN(LN2(y)))

	copy(e.cacheX, x)
	out := make([]T, len(e.bufF))
	copy(out, e.bufF)
	return out
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

// backwardPostNorm reverses the post-norm forward chain using per-position
// BackwardSeq for correct multi-position LN gradient accumulation (T-18A05).
func (e *EncoderBlock[T]) backwardPostNorm(upstream []T) []T {
	seqLen := e.Cfg.SeqLen

	// reverse LN2 — BackwardSeq uses per-position xHat/invSd cached by ForwardSeq
	dBufF := e.Norm2.BackwardSeq(upstream, seqLen)

	// reverse residual 2: gradient flows to both FFN path and z1 path
	dFFNOut := make([]T, len(dBufF))
	copy(dFFNOut, dBufF)
	dZ1fromRes2 := dBufF

	// reverse Drop2 — BackwardMask re-applies the stored retain mask (TRANS-C7)
	dFFNOut = e.Drop2.BackwardMask(dFFNOut)
	// reverse FFN (full sequence)
	dZ1fromFFN := e.FFN.Backward(dFFNOut)

	// combine z1 gradients
	dZ1 := make([]T, len(dZ1fromRes2))
	copy(dZ1, dZ1fromRes2)
	addInPlace(dZ1, dZ1fromFFN)

	// reverse LN1 — BackwardSeq uses per-position xHat/invSd cached by ForwardSeq
	dBufZ := e.Norm1.BackwardSeq(dZ1, seqLen)

	// reverse residual 1: gradient flows to both Attn path and input path
	dAttnOut := make([]T, len(dBufZ))
	copy(dAttnOut, dBufZ)
	dXfromRes1 := dBufZ

	// reverse Drop1 — BackwardMask re-applies the stored retain mask (TRANS-C7)
	dAttnOut = e.Drop1.BackwardMask(dAttnOut)
	// reverse Attn (full sequence)
	dXfromAttn := e.Attn.Backward(dAttnOut)

	// combine input gradients
	dx := make([]T, len(dXfromRes1))
	copy(dx, dXfromRes1)
	addInPlace(dx, dXfromAttn)

	return dx
}

// backwardPreNorm reverses the pre-norm forward chain (TRANS-3 + TRANS-8).
//
//	upstream → residual₂ split → FFN → LN₂ ┐
//	                                         ├→ combine → residual₁ split → Attn → LN₁ ┐
//	                  residual₂ direct ──────┘                                           ├→ dx
//	                                                        residual₁ direct ────────────┘
func (e *EncoderBlock[T]) backwardPreNorm(upstream []T) []T {
	seqLen := e.Cfg.SeqLen

	// reverse second residual: z = y + Drop2(FFN(LN2(y)))
	dFFNOut := make([]T, len(upstream))
	copy(dFFNOut, upstream)
	dYfromRes2 := upstream // identity path through residual

	// reverse Drop2 — BackwardMask re-applies the stored retain mask (TRANS-C7)
	dFFNOut = e.Drop2.BackwardMask(dFFNOut)
	// reverse FFN
	dLN2Out := e.FFN.Backward(dFFNOut)

	// reverse LN2 — BackwardSeq uses per-position xHat from ForwardSeq(y)
	dYfromLN2 := e.Norm2.BackwardSeq(dLN2Out, seqLen)

	// combine y gradients
	dY := make([]T, len(dYfromRes2))
	copy(dY, dYfromRes2)
	addInPlace(dY, dYfromLN2)

	// reverse first residual: y = x + Drop1(Attn(LN1(x)))
	dAttnOut := make([]T, len(dY))
	copy(dAttnOut, dY)
	dXfromRes1 := dY // identity path through residual

	// reverse Drop1 — BackwardMask re-applies the stored retain mask (TRANS-C7)
	dAttnOut = e.Drop1.BackwardMask(dAttnOut)
	// reverse Attn
	dLN1Out := e.Attn.Backward(dAttnOut)

	// reverse LN1 — BackwardSeq uses per-position xHat from ForwardSeq(x)
	dXfromLN1 := e.Norm1.BackwardSeq(dLN1Out, seqLen)

	// combine input gradients
	dx := make([]T, len(dXfromRes1))
	copy(dx, dXfromRes1)
	addInPlace(dx, dXfromLN1)

	return dx
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

// MarshalJSON serialises the EncoderBlock to a JSON envelope per TRANS-9.
// Each child is embedded as a pre-marshalled raw object so child schema
// evolution stays decoupled from the envelope format.
//
// AI-Meta:
//   - Purpose: JSON persistence for EncoderBlock checkpoint and round-trip.
//   - Related: [EncoderBlock.UnmarshalJSON], [TransformerConfig].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) MarshalJSON() ([]byte, error) {
	attnRaw, err := json.Marshal(e.Attn)
	if err != nil {
		return nil, fmt.Errorf("EncoderBlock.MarshalJSON Attn: %w", err)
	}
	norm1Raw, err := json.Marshal(e.Norm1)
	if err != nil {
		return nil, fmt.Errorf("EncoderBlock.MarshalJSON Norm1: %w", err)
	}
	norm2Raw, err := json.Marshal(e.Norm2)
	if err != nil {
		return nil, fmt.Errorf("EncoderBlock.MarshalJSON Norm2: %w", err)
	}
	ffnRaw, err := json.Marshal(e.FFN)
	if err != nil {
		return nil, fmt.Errorf("EncoderBlock.MarshalJSON FFN: %w", err)
	}
	return json.Marshal(&struct {
		Type   string               `json:"Type"`
		Attn   json.RawMessage      `json:"Attn"`
		Norm1  json.RawMessage      `json:"Norm1"`
		Norm2  json.RawMessage      `json:"Norm2"`
		FFN    json.RawMessage      `json:"FFN"`
		Config TransformerConfig[T] `json:"Config"`
	}{
		Type:   "transformer.EncoderBlock",
		Config: e.Cfg,
		Attn:   attnRaw,
		Norm1:  norm1Raw,
		Norm2:  norm2Raw,
		FFN:    ffnRaw,
	})
}

// UnmarshalJSON restores an EncoderBlock from a JSON envelope. Validates the
// Type tag, reconstructs all children, rebuilds Dropout from DropoutRate, and
// pre-allocates forward-cache buffers so Forward is ready without Init.
//
// AI-Meta:
//   - Purpose: JSON restoration for EncoderBlock checkpoint round-trip.
//   - Related: [EncoderBlock.MarshalJSON], [NewEncoderBlock].
//   - Stability: Experimental.
func (e *EncoderBlock[T]) UnmarshalJSON(data []byte) error {
	aux := &struct {
		Type   string               `json:"Type"`
		Attn   json.RawMessage      `json:"Attn"`
		Norm1  json.RawMessage      `json:"Norm1"`
		Norm2  json.RawMessage      `json:"Norm2"`
		FFN    json.RawMessage      `json:"FFN"`
		Config TransformerConfig[T] `json:"Config"`
	}{}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if aux.Type != "transformer.EncoderBlock" {
		return fmt.Errorf("EncoderBlock.UnmarshalJSON: unexpected Type %q", aux.Type)
	}
	e.Cfg = aux.Config

	e.Attn = &attention.MultiHeadAttention[T]{}
	if err := json.Unmarshal(aux.Attn, e.Attn); err != nil {
		return fmt.Errorf("EncoderBlock.UnmarshalJSON Attn: %w", err)
	}
	e.Norm1 = &norm.LayerNorm[T]{}
	if err := json.Unmarshal(aux.Norm1, e.Norm1); err != nil {
		return fmt.Errorf("EncoderBlock.UnmarshalJSON Norm1: %w", err)
	}
	e.Norm2 = &norm.LayerNorm[T]{}
	if err := json.Unmarshal(aux.Norm2, e.Norm2); err != nil {
		return fmt.Errorf("EncoderBlock.UnmarshalJSON Norm2: %w", err)
	}
	e.FFN = &ffn[T]{}
	if err := json.Unmarshal(aux.FFN, e.FFN); err != nil {
		return fmt.Errorf("EncoderBlock.UnmarshalJSON FFN: %w", err)
	}

	p := retentionProb(e.Cfg.DropoutRate)
	e.Drop1 = regularizer.NewDropout[T](p)
	e.Drop2 = regularizer.NewDropout[T](p)

	size := e.Cfg.SeqLen * e.Cfg.Dmodel
	e.bufAttn = make([]T, size)
	e.bufZ = make([]T, size)
	e.bufF = make([]T, size)
	e.cacheX = make([]T, size)
	e.cacheZ1 = make([]T, size)
	return nil
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
