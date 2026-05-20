// Package transformer provides Transformer block primitives for the GoNN
// neural network library.
//
// The package exports three composable types — [EncoderBlock], [DecoderBlock],
// and [Stack] — satisfying TRANS-1..10 invariants from l1-transformer-block and
// their implementation contract in l2-transformer-impl.
//
// All three types implement [layer.Layer] so they can be appended to the
// ConvPrefix slot via [nn.WithEncoderBlock], [nn.WithDecoderBlock],
// [nn.WithEncoderStack], and [nn.WithDecoderStack] options without any new
// Config field or compile.go change.
//
// Reuses existing stable primitives:
//   - [attention.MultiHeadAttention] for self-attention (Phase 17)
//   - [norm.LayerNorm] for the two per-block normalization layers (Phase 10)
//   - [regularizer.Dropout] for Drop1/Drop2 positions (Phase 6)
//   - [activation.Activation] dispatcher for the position-wise FFN (Phase 3)
//
// AI-Meta:
//   - Purpose: Composite Transformer-block primitives — encoder / decoder / stack.
//   - Tier: L2-impl.
//   - Stability: Experimental.
package transformer
