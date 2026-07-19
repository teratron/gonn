package embedding

import (
	"encoding/json"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/utils"
)

// EmbeddingStack composes a [TokenEmbedding] followed by a [PositionalEncoding]
// into a single [layer.IDLayer]. It is the canonical first layer for sequence
// models: token IDs → dense embeddings → positional offsets added.
//
// Forward path:  IDs → TokenEmbedding.ForwardIDs → PositionalEncoding.Forward
// Backward path: upstream → PositionalEncoding.Backward → TokenEmbedding.Backward
//
// AI-Meta:
//   - Purpose: Compose TokenEmbedding + PositionalEncoding as a single IDLayer (EMB-10).
//   - Usage: nn.WithEmbeddingStack or direct construction; satisfies layer.IDLayer[T].
//   - Related: [TokenEmbedding], [PositionalEncoding], [layer.IDLayer].
//   - Stability: Stable.
type EmbeddingStack[T utils.Float] struct {
	Token      *TokenEmbedding[T]     `json:"token"`
	Positional *PositionalEncoding[T] `json:"positional"`
}

// compile-time assertions.
var _ layer.Layer[float32] = (*EmbeddingStack[float32])(nil)
var _ layer.Layer[float64] = (*EmbeddingStack[float64])(nil)
var _ layer.IDLayer[float32] = (*EmbeddingStack[float32])(nil)
var _ layer.IDLayer[float64] = (*EmbeddingStack[float64])(nil)

// NewEmbeddingStack constructs an EmbeddingStack with the given vocabulary size,
// sequence length, model dimension, and positional encoding mode. Call Init before use.
func NewEmbeddingStack[T utils.Float](vocabSize, seqLen, dmodel int, mode PositionalMode) *EmbeddingStack[T] {
	return &EmbeddingStack[T]{
		Token:      NewTokenEmbedding[T](vocabSize, seqLen, dmodel),
		Positional: NewPositionalEncoding[T](seqLen, dmodel, mode),
	}
}

// Init initialises both sub-layers using rng.
func (es *EmbeddingStack[T]) Init(rng *rand.Rand) {
	es.Token.Init(rng)
	es.Positional.Init(rng)
}

// InputSize returns SeqLen — the number of integer IDs expected per call.
func (es *EmbeddingStack[T]) InputSize() int { return es.Token.SeqLen }

// OutputSize returns SeqLen*Dmodel — flat output size.
func (es *EmbeddingStack[T]) OutputSize() int { return es.Token.OutputSize() }

// GradSlots delegates to TokenEmbedding (primary learned weights).
// PositionalEncoding gradients are accessed directly via es.Positional.GradSlots().
func (es *EmbeddingStack[T]) GradSlots() (gradW, gradB []T) {
	return es.Token.GradSlots()
}

// ApplyGradSGD updates both sub-layers: the sparse token table and the
// learnable positional table (a no-op for the sinusoidal variant). Each
// sub-layer resets its own accumulator.
func (es *EmbeddingStack[T]) ApplyGradSGD(lr T) {
	es.Token.ApplyGradSGD(lr)
	es.Positional.ApplyGradSGD(lr)
}

// ForwardIDs runs the full embedding pipeline: lookup → positional encoding.
func (es *EmbeddingStack[T]) ForwardIDs(ids []int) ([]T, error) {
	embedded, err := es.Token.ForwardIDs(ids)
	if err != nil {
		return nil, err
	}
	return es.Positional.Forward(embedded), nil
}

// Forward converts x to integer IDs and delegates to ForwardIDs.
func (es *EmbeddingStack[T]) Forward(x []T) []T {
	ids := make([]int, len(x))
	for i, v := range x {
		ids[i] = int(v)
	}
	out, _ := es.ForwardIDs(ids)
	return out
}

// Backward propagates upstream through PositionalEncoding first (returns dEmbedded),
// then into TokenEmbedding (accumulates sparse table gradients, returns nil).
func (es *EmbeddingStack[T]) Backward(upstream []T) []T {
	dEmbedded := es.Positional.Backward(upstream)
	return es.Token.Backward(dEmbedded)
}

// embeddingStackJSON is the on-wire representation.
type embeddingStackJSON[T utils.Float] struct {
	Token      *TokenEmbedding[T]     `json:"token"`
	Positional *PositionalEncoding[T] `json:"positional"`
	Type       string                 `json:"type"`
}

// MarshalJSON serialises both sub-layers.
func (es *EmbeddingStack[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(embeddingStackJSON[T]{
		Type:       "EmbeddingStack",
		Token:      es.Token,
		Positional: es.Positional,
	})
}

// UnmarshalJSON restores both sub-layers.
func (es *EmbeddingStack[T]) UnmarshalJSON(data []byte) error {
	var j embeddingStackJSON[T]
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	es.Token = j.Token
	es.Positional = j.Positional
	return nil
}
