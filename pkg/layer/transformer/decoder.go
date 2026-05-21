package transformer

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/layer/attention"
	"github.com/teratron/gonn/pkg/layer/norm"
	"github.com/teratron/gonn/pkg/regularizer"
	"github.com/teratron/gonn/pkg/utils"
)

// DecoderBlock implements a single Transformer decoder layer with causal
// (autoregressive) self-attention:
//
//	Post-norm:  y = LN₂(Z + FFN(LN₁(X + Attn(X))))
//	Pre-norm:   Y = X + Attn(LN₁(X)); Z = Y + FFN(LN₂(Y))  (TRANS-3)
//
// The only difference from EncoderBlock is that the inner MHA is built with
// Causal=true, which applies a causal mask each Forward pass (TRANS-4 / TRANS-C2).
//
// AI-Meta:
//   - Purpose: Single Transformer decoder layer; causal self-attention blocks future positions.
//   - Usage: blk := transformer.NewDecoderBlock[float32](cfg); blk.Init(rng).
//   - Concurrency: NotSafe; Forward/Backward mutate per-call cache buffers.
//   - Related: [EncoderBlock], [Stack], [TransformerConfig], [layer.Layer].
//   - Stability: Experimental.
type DecoderBlock[T utils.Float] struct {
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

// NewDecoderBlock constructs a DecoderBlock for the given config.
// Applies the same defaults as NewEncoderBlock (Activation → ReLU, Dff → 4·Dmodel).
// The inner MHA is constructed with Causal=true to enforce autoregressive masking
// (TRANS-4). Call Init(rng) before use.
//
// AI-Meta:
//   - Purpose: Allocate a DecoderBlock with causal self-attention; weights zeroed until Init.
//   - Usage: blk := transformer.NewDecoderBlock[float64](cfg); blk.Init(rng).
//   - Related: [DecoderBlock.Init], [TransformerConfig], [EncoderBlock].
//   - Stability: Experimental.
func NewDecoderBlock[T utils.Float](cfg TransformerConfig[T]) *DecoderBlock[T] {
	if cfg.Activation == 0 {
		cfg.Activation = defaultActivation
	}
	if cfg.Dff == 0 {
		cfg.Dff = 4 * cfg.Dmodel
	}
	p := retentionProb(cfg.DropoutRate)
	return &DecoderBlock[T]{
		Cfg:   cfg,
		Attn:  attention.NewMultiHeadAttention[T](cfg.SeqLen, cfg.Dmodel, cfg.NumHeads, true), // causal=true
		Norm1: norm.NewLayerNorm[T](cfg.Dmodel),
		Norm2: norm.NewLayerNorm[T](cfg.Dmodel),
		FFN:   newFFN[T](cfg.SeqLen, cfg.Dmodel, cfg.Dff, cfg.Activation),
		Drop1: regularizer.NewDropout[T](p),
		Drop2: regularizer.NewDropout[T](p),
	}
}

// Init Xavier-initializes all weight matrices and pre-allocates forward/backward
// cache buffers. Each child receives an independent RNG stream forked from rng
// (TRANS-C8 seed isolation).
//
// AI-Meta:
//   - Purpose: Initialize weights and pre-allocate caches; must be called before Forward.
//   - Concurrency: NotSafe.
//   - Related: [DecoderBlock], [NewDecoderBlock].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) Init(rng *rand.Rand) {
	s0, s1 := rng.Uint64(), rng.Uint64()
	d.Attn.Init(rand.New(rand.NewPCG(s0, s0^0x9E3779B9)))
	d.FFN.Init(rand.New(rand.NewPCG(s1, s1^0x9E3779B9)))
	size := d.Cfg.SeqLen * d.Cfg.Dmodel
	d.bufAttn = make([]T, size)
	d.bufZ = make([]T, size)
	d.bufF = make([]T, size)
	d.cacheX = make([]T, size)
	d.cacheZ1 = make([]T, size)
}

// SetTraining controls whether Dropout is applied during Forward/Backward.
func (d *DecoderBlock[T]) SetTraining(training bool) {
	d.training = training
}

// Forward applies the post-norm (default) or pre-norm (when Cfg.PreNorm=true)
// decoder sub-layers and returns the output tensor of shape SeqLen*Dmodel.
//
// AI-Meta:
//   - Purpose: Decoder forward pass with causal masking; preserves shape (TRANS-1).
//   - Concurrency: NotSafe; mutates cache buffers.
//   - Related: [DecoderBlock.Backward], [DecoderBlock.Init].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) Forward(x []T) []T {
	// Lazy-allocate cache buffers if Init has not been called yet.
	if d.bufZ == nil {
		size := d.Cfg.SeqLen * d.Cfg.Dmodel
		d.bufAttn = make([]T, size)
		d.bufZ = make([]T, size)
		d.bufF = make([]T, size)
		d.cacheX = make([]T, size)
		d.cacheZ1 = make([]T, size)
	}
	if d.Cfg.PreNorm {
		return d.forwardPreNorm(x)
	}
	return d.forwardPostNorm(x)
}

func (d *DecoderBlock[T]) forwardPostNorm(x []T) []T {
	seqLen := d.Cfg.SeqLen

	attnOut := d.Attn.Forward(x)
	attnOut = d.Drop1.ApplyMask(attnOut, d.training)
	copy(d.bufZ, x)
	addInPlace(d.bufZ, attnOut)
	copy(d.cacheZ1, d.Norm1.ForwardSeq(d.bufZ, seqLen))

	ffnOut := d.FFN.Forward(d.cacheZ1)
	ffnOut = d.Drop2.ApplyMask(ffnOut, d.training)
	copy(d.bufF, d.cacheZ1)
	addInPlace(d.bufF, ffnOut)
	out := d.Norm2.ForwardSeq(d.bufF, seqLen)

	copy(d.cacheX, x)
	return out
}

func (d *DecoderBlock[T]) forwardPreNorm(x []T) []T {
	seqLen := d.Cfg.SeqLen

	ln1Out := d.Norm1.ForwardSeq(x, seqLen)
	attnOut := d.Attn.Forward(ln1Out)
	attnOut = d.Drop1.ApplyMask(attnOut, d.training)
	copy(d.bufZ, x)
	addInPlace(d.bufZ, attnOut)
	copy(d.cacheZ1, d.bufZ)

	ln2Out := d.Norm2.ForwardSeq(d.cacheZ1, seqLen)
	ffnOut := d.FFN.Forward(ln2Out)
	ffnOut = d.Drop2.ApplyMask(ffnOut, d.training)
	copy(d.bufF, d.cacheZ1)
	addInPlace(d.bufF, ffnOut)

	copy(d.cacheX, x)
	out := make([]T, len(d.bufF))
	copy(out, d.bufF)
	return out
}

// Backward computes ∂L/∂x given the upstream gradient and accumulates
// parameter gradients in all children's GradSlots buffers.
//
// AI-Meta:
//   - Purpose: Decoder backward pass; returns ∂L/∂x.
//   - Concurrency: NotSafe.
//   - Related: [DecoderBlock.Forward], [DecoderBlock.ApplyGradSGD].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) Backward(upstream []T) []T {
	if d.Cfg.PreNorm {
		return d.backwardPreNorm(upstream)
	}
	return d.backwardPostNorm(upstream)
}

func (d *DecoderBlock[T]) backwardPostNorm(upstream []T) []T {
	seqLen := d.Cfg.SeqLen

	dBufF := d.Norm2.BackwardSeq(upstream, seqLen)

	dFFNOut := make([]T, len(dBufF))
	copy(dFFNOut, dBufF)
	dZ1fromRes2 := dBufF

	dFFNOut = d.Drop2.BackwardMask(dFFNOut)
	dZ1fromFFN := d.FFN.Backward(dFFNOut)

	dZ1 := make([]T, len(dZ1fromRes2))
	copy(dZ1, dZ1fromRes2)
	addInPlace(dZ1, dZ1fromFFN)

	dBufZ := d.Norm1.BackwardSeq(dZ1, seqLen)

	dAttnOut := make([]T, len(dBufZ))
	copy(dAttnOut, dBufZ)
	dXfromRes1 := dBufZ

	dAttnOut = d.Drop1.BackwardMask(dAttnOut)
	dXfromAttn := d.Attn.Backward(dAttnOut)

	dx := make([]T, len(dXfromRes1))
	copy(dx, dXfromRes1)
	addInPlace(dx, dXfromAttn)

	return dx
}

func (d *DecoderBlock[T]) backwardPreNorm(upstream []T) []T {
	seqLen := d.Cfg.SeqLen

	dFFNOut := make([]T, len(upstream))
	copy(dFFNOut, upstream)
	dYfromRes2 := upstream

	dFFNOut = d.Drop2.BackwardMask(dFFNOut)
	dLN2Out := d.FFN.Backward(dFFNOut)

	dYfromLN2 := d.Norm2.BackwardSeq(dLN2Out, seqLen)

	dY := make([]T, len(dYfromRes2))
	copy(dY, dYfromRes2)
	addInPlace(dY, dYfromLN2)

	dAttnOut := make([]T, len(dY))
	copy(dAttnOut, dY)
	dXfromRes1 := dY

	dAttnOut = d.Drop1.BackwardMask(dAttnOut)
	dLN1Out := d.Attn.Backward(dAttnOut)

	dXfromLN1 := d.Norm1.BackwardSeq(dLN1Out, seqLen)

	dx := make([]T, len(dXfromRes1))
	copy(dx, dXfromRes1)
	addInPlace(dx, dXfromLN1)

	return dx
}

// GradSlots satisfies layer.Layer[T].
func (d *DecoderBlock[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// InputSize returns SeqLen * Dmodel (TRANS-1).
func (d *DecoderBlock[T]) InputSize() int { return d.Cfg.SeqLen * d.Cfg.Dmodel }

// OutputSize returns SeqLen * Dmodel (TRANS-1).
func (d *DecoderBlock[T]) OutputSize() int { return d.Cfg.SeqLen * d.Cfg.Dmodel }

// ApplyGradSGD fans the inline SGD step to every child with learnable parameters.
//
// AI-Meta:
//   - Purpose: Drive inline SGD update across all child parameter groups.
//   - Related: [DecoderBlock.Backward].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) ApplyGradSGD(lr T) {
	d.Attn.ApplyGradSGD(lr)
	d.Norm1.ApplyGradSGD(lr)
	d.Norm2.ApplyGradSGD(lr)
	d.FFN.ApplyGradSGD(lr)
}

// SetPaddingMask forwards the padding mask to the inner MHA (TRANS-10).
//
// AI-Meta:
//   - Purpose: Forward the padding mask to the inner MultiHeadAttention layer.
//   - Related: [attention.MaskedLayer], [attention.MultiHeadAttention.SetPaddingMask].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) SetPaddingMask(mask []bool) {
	d.Attn.SetPaddingMask(mask)
}

// MarshalJSON serialises the DecoderBlock to a JSON envelope per TRANS-9.
//
// AI-Meta:
//   - Purpose: JSON persistence for DecoderBlock checkpoint and round-trip.
//   - Related: [DecoderBlock.UnmarshalJSON], [TransformerConfig].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) MarshalJSON() ([]byte, error) {
	attnRaw, err := json.Marshal(d.Attn)
	if err != nil {
		return nil, fmt.Errorf("DecoderBlock.MarshalJSON Attn: %w", err)
	}
	norm1Raw, err := json.Marshal(d.Norm1)
	if err != nil {
		return nil, fmt.Errorf("DecoderBlock.MarshalJSON Norm1: %w", err)
	}
	norm2Raw, err := json.Marshal(d.Norm2)
	if err != nil {
		return nil, fmt.Errorf("DecoderBlock.MarshalJSON Norm2: %w", err)
	}
	ffnRaw, err := json.Marshal(d.FFN)
	if err != nil {
		return nil, fmt.Errorf("DecoderBlock.MarshalJSON FFN: %w", err)
	}
	return json.Marshal(&struct {
		Type   string               `json:"Type"`
		Attn   json.RawMessage      `json:"Attn"`
		Norm1  json.RawMessage      `json:"Norm1"`
		Norm2  json.RawMessage      `json:"Norm2"`
		FFN    json.RawMessage      `json:"FFN"`
		Config TransformerConfig[T] `json:"Config"`
	}{
		Type:   "transformer.DecoderBlock",
		Config: d.Cfg,
		Attn:   attnRaw,
		Norm1:  norm1Raw,
		Norm2:  norm2Raw,
		FFN:    ffnRaw,
	})
}

// UnmarshalJSON restores a DecoderBlock from a JSON envelope. Validates the
// Type tag, reconstructs all children, rebuilds Dropout from DropoutRate, and
// pre-allocates forward-cache buffers so Forward is ready without Init.
//
// AI-Meta:
//   - Purpose: JSON restoration for DecoderBlock checkpoint round-trip.
//   - Related: [DecoderBlock.MarshalJSON], [NewDecoderBlock].
//   - Stability: Experimental.
func (d *DecoderBlock[T]) UnmarshalJSON(data []byte) error {
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
	if aux.Type != "transformer.DecoderBlock" {
		return fmt.Errorf("DecoderBlock.UnmarshalJSON: unexpected Type %q", aux.Type)
	}
	d.Cfg = aux.Config

	d.Attn = &attention.MultiHeadAttention[T]{}
	if err := json.Unmarshal(aux.Attn, d.Attn); err != nil {
		return fmt.Errorf("DecoderBlock.UnmarshalJSON Attn: %w", err)
	}
	d.Norm1 = &norm.LayerNorm[T]{}
	if err := json.Unmarshal(aux.Norm1, d.Norm1); err != nil {
		return fmt.Errorf("DecoderBlock.UnmarshalJSON Norm1: %w", err)
	}
	d.Norm2 = &norm.LayerNorm[T]{}
	if err := json.Unmarshal(aux.Norm2, d.Norm2); err != nil {
		return fmt.Errorf("DecoderBlock.UnmarshalJSON Norm2: %w", err)
	}
	d.FFN = &ffn[T]{}
	if err := json.Unmarshal(aux.FFN, d.FFN); err != nil {
		return fmt.Errorf("DecoderBlock.UnmarshalJSON FFN: %w", err)
	}

	p := retentionProb(d.Cfg.DropoutRate)
	d.Drop1 = regularizer.NewDropout[T](p)
	d.Drop2 = regularizer.NewDropout[T](p)

	size := d.Cfg.SeqLen * d.Cfg.Dmodel
	d.bufAttn = make([]T, size)
	d.bufZ = make([]T, size)
	d.bufF = make([]T, size)
	d.cacheX = make([]T, size)
	d.cacheZ1 = make([]T, size)
	return nil
}

// Compile-time interface assertions.
var (
	_ layer.Layer[float32]           = (*DecoderBlock[float32])(nil)
	_ layer.Layer[float64]           = (*DecoderBlock[float64])(nil)
	_ attention.MaskedLayer[float32] = (*DecoderBlock[float32])(nil)
	_ attention.MaskedLayer[float64] = (*DecoderBlock[float64])(nil)
	_ Block[float32]                 = (*DecoderBlock[float32])(nil)
	_ Block[float64]                 = (*DecoderBlock[float64])(nil)
)
