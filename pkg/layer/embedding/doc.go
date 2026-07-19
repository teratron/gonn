// Package embedding provides token and positional embedding layers for the GoNN library.
//
// Implements [l1-embedding] EMB-1..EMB-10 with three concrete types:
//
//   - [TokenEmbedding]: maps integer token IDs to dense vectors via a learnable
//     lookup table; supports sparse gradient accumulation (EMB-9).
//   - [PositionalEncoding]: adds position information to token embeddings via
//     fixed sinusoidal encoding (Vaswani et al. 2017) or learnable position
//     vectors controlled by [PositionalMode].
//   - [EmbeddingStack]: composes TokenEmbedding + PositionalEncoding into a
//     single [layer.IDLayer] suitable for use as the first layer of a network.
//
// Usage in a network (pkg/nn functional options):
//
//	nn.New[float64](
//	    nn.WithEmbedding[float64](vocabSize, dmodel),
//	    nn.WithPositionalEncoding[float64](seqLen, dmodel, embedding.Sinusoidal),
//	    nn.WithMultiHeadAttention[float64](seqLen, dmodel, numHeads),
//	    nn.WithOutput[float64](numClasses, activation.SOFTMAX),
//	)
//
// Gradient flow: TokenEmbedding.Backward accumulates sparse row gradients via
// sparseGrad (see sparse.go), enabling O(unique IDs × Dmodel) weight updates
// per step rather than O(VocabSize × Dmodel).
//
// AI-Meta:
//   - Purpose: Token and positional embedding layers implementing EMB-1..EMB-10 for GoNN.
//   - Usage: Use WithEmbedding/WithPositionalEncoding options when building a NN.
//   - Concurrency: NotSafe; Forward/Backward mutate per-call state.
//   - Related: [TokenEmbedding], [PositionalEncoding], [EmbeddingStack], [layer.IDLayer].
//   - Stability: Stable.
package embedding
