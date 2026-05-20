// Package attention provides the multi-head attention layer for the GoNN library.
//
// Implements [l1-attention] ATT-1..ATT-10 with a single canonical struct
// [MultiHeadAttention] covering all three L1 conceptual variants (Attention,
// SelfAttention, MultiHeadAttention) via the NumHeads field:
//
//   - NumHeads = 1: single-head attention (equivalent to SelfAttention)
//   - NumHeads > 1: multi-head attention with independent per-head projections
//
// Usage in a network (pkg/nn functional options):
//
//	nn.New[float64](
//	    nn.WithInput[float64](seqLen * dmodel),
//	    nn.WithMultiHeadAttention[float64](seqLen, dmodel, numHeads),
//	    nn.WithCausalAttention[float64](),
//	    nn.WithOutput[float64](numClasses, activation.SOFTMAX),
//	)
//
// Softmax helpers (cell.go) are package-private and correspond directly to
// ATT-3 / ATT-5 / ATT-6 / ATT-7. They are validated independently from the
// attention layer forward/backward tests.
//
// AI-Meta:
//   - Purpose: Multi-head attention layer implementing ATT-1..ATT-10 for sequence modelling in GoNN.
//   - Usage: Use WithAttention/WithMultiHeadAttention/WithCausalAttention options when building a NN.
//   - Concurrency: NotSafe; Forward/Backward mutate per-call caches — one goroutine per network.
//   - Related: [MultiHeadAttention], [MaskedLayer], [conv.Layer].
//   - Stability: Stable.
package attention
