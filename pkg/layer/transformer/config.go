package transformer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

// Mode selects whether a block or stack is used in encoder (bidirectional) or
// decoder (causal / autoregressive) mode. The mode determines the Causal flag
// on the inner MultiHeadAttention.
//
// AI-Meta:
//   - Purpose: Enum controlling encoder-vs-decoder wiring of the inner attention layer.
//   - Usage: transformer.NewStack(cfg, transformer.EncoderMode, 6).
//   - Related: [TransformerConfig], [EncoderBlock], [DecoderBlock], [Stack].
//   - Stability: Experimental.
type Mode uint8

const (
	// EncoderMode uses bidirectional self-attention (Causal:false).
	EncoderMode Mode = iota
	// DecoderMode uses causal (upper-triangular masked) self-attention (Causal:true).
	DecoderMode
)

// TransformerConfig holds all construction-time hyperparameters shared by
// [EncoderBlock], [DecoderBlock], and [Stack].
//
// Callers that omit Dff receive 4·Dmodel applied by the constructor (TRANS-C5).
// Callers that omit Activation receive ReLU applied by the constructor (TRANS-C4);
// the zero-value of [activation.Type] is ELISH, not ReLU.
// PreNorm selects pre-norm (LN before sub-layer) vs post-norm (LN after residual) wiring
// at construction time — see TRANS-3 and TRANS-C6.
//
// AI-Meta:
//   - Purpose: Construction-time hyperparameters for transformer block primitives.
//   - Usage: cfg := transformer.TransformerConfig[float32]{SeqLen:16, Dmodel:64, NumHeads:4}.
//   - Related: [EncoderBlock], [DecoderBlock], [Stack], [Mode].
//   - Concurrency: Immutable after construction; safe to share across blocks.
//   - Stability: Experimental.
type TransformerConfig[T utils.Float] struct {
	DropoutRate T
	SeqLen      int
	Dmodel      int
	NumHeads    int
	Dff         int
	PreNorm     bool
	Activation  activation.Type
}

// applyActivationInPlace applies the named activation function to every element
// of x in-place using the scalar [activation.Activation] dispatcher.
// Avoids allocations by mutating x directly.
//
// AI-Meta:
//   - Purpose: Slice-wise activation helper for the FFN hidden-layer non-linearity.
//   - Concurrency: NotSafe; mutates x.
//   - Related: [activation.Activation], [TransformerConfig.Activation].
func applyActivationInPlace[T utils.Float](mode activation.Type, x []T) {
	for i, v := range x {
		x[i] = activation.Activation(v, mode)
	}
}
