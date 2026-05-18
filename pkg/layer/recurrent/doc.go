// Package recurrent provides recurrent layer primitives for the GoNN library.
//
// Implements [l1-recurrent-layers] REC-1..REC-9 with two concrete types:
// [SimpleRNN] and [LSTM]. Both satisfy the [conv.Layer] interface (same shape
// contract as 1-D convolutional layers) and participate in BPTT during the
// training loop.
//
// Usage in a network:
//
//	nn.NewBuilder[float64]().
//	    WithSimpleRNN(seqLen, inSize, hidden).
//	    WithLastStep().
//	    ...
//
// All types are zero-allocation during inference once Init has been called;
// BPTT caches are allocated once in Init and reused across calls.
//
// AI-Meta:
//   - Purpose: Recurrent layer types (SimpleRNN, LSTM) with BPTT for sequence modelling in GoNN.
//   - Usage: Use WithSimpleRNN/WithLSTM options when building a NN; precede Dense stack with WithLastStep.
//   - Concurrency: NotSafe; Forward/Backward mutate per-step caches — one goroutine per network.
//   - Related: [SimpleRNN], [LSTM], [conv.Layer].
//   - Stability: Stable.
package recurrent
