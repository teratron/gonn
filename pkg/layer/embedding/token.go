package embedding

import (
	"encoding/json"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/utils"
)

// TokenEmbedding maps integer token IDs to dense vectors using a learnable
// lookup table of shape [VocabSize × Dmodel]. It implements [layer.IDLayer]
// so the network topology compiler can route integer-ID data flow (EMB-1..EMB-4).
//
// Gradient flow: Backward accumulates sparse row gradients via [sparseGrad]
// (EMB-9). The caller retrieves them with GradTable / GradSlots and applies
// them via an optimizer's sparse update path (T-17B06).
//
// AI-Meta:
//   - Purpose: Learnable token lookup table implementing EMB-1..EMB-9.
//   - Usage: Use via EmbeddingStack or directly with the IDLayer interface.
//   - Related: [EmbeddingStack], [sparseGrad], [layer.IDLayer].
//   - Stability: Stable.
type TokenEmbedding[T utils.Float] struct {
	Table     []T `json:"table"`
	lastIDs   []int
	lastOut   []T
	gradTable sparseGrad[T]
	VocabSize int `json:"vocabSize"`
	SeqLen    int `json:"seqLen"`
	Dmodel    int `json:"dmodel"`
}

// compile-time assertions.
var _ layer.Layer[float32] = (*TokenEmbedding[float32])(nil)
var _ layer.Layer[float64] = (*TokenEmbedding[float64])(nil)
var _ layer.IDLayer[float32] = (*TokenEmbedding[float32])(nil)
var _ layer.IDLayer[float64] = (*TokenEmbedding[float64])(nil)

// NewTokenEmbedding creates an uninitialised TokenEmbedding. Call Init before use.
func NewTokenEmbedding[T utils.Float](vocabSize, seqLen, dmodel int) *TokenEmbedding[T] {
	if vocabSize < 1 {
		vocabSize = 1
	}
	if seqLen < 1 {
		seqLen = 1
	}
	if dmodel < 1 {
		dmodel = 1
	}
	return &TokenEmbedding[T]{
		VocabSize: vocabSize,
		SeqLen:    seqLen,
		Dmodel:    dmodel,
	}
}

// Init initialises the lookup table with XavierUniform values and pre-allocates
// caches and the sparse gradient accumulator.
func (te *TokenEmbedding[T]) Init(rng *rand.Rand) {
	te.Table = make([]T, te.VocabSize*te.Dmodel)
	for i := range te.Table {
		te.Table[i] = utils.XavierUniform[T](rng, te.VocabSize, te.Dmodel)
	}
	te.gradTable = newSparseGrad[T](te.VocabSize, te.Dmodel)
	te.lastIDs = make([]int, te.SeqLen)
	te.lastOut = make([]T, te.SeqLen*te.Dmodel)
}

// InputSize returns SeqLen — the number of integer IDs consumed per call.
func (te *TokenEmbedding[T]) InputSize() int { return te.SeqLen }

// OutputSize returns SeqLen*Dmodel — the flat size of the output embedding matrix.
func (te *TokenEmbedding[T]) OutputSize() int { return te.SeqLen * te.Dmodel }

// GradSlots returns (gradTable.Buf, nil). The dense Buf holds accumulated sparse
// gradients; the optimizer applies them row-by-row via GradTable().
// The second slot is nil because TokenEmbedding has no bias vector.
func (te *TokenEmbedding[T]) GradSlots() (gradW, gradB []T) {
	return te.gradTable.Buf, nil
}

// ApplyGradSGD applies sparse SGD to every touched row: Table[r] -= lr * grad[r],
// then resets the gradient accumulator. Called by applyTokenEmbeddingSGD in
// pkg/nn/train.go after the backward pass (mirrors MultiHeadAttention.ApplyGradSGD).
func (te *TokenEmbedding[T]) ApplyGradSGD(lr T) {
	te.gradTable.iter(func(row int, grad []T) {
		base := row * te.Dmodel
		for i, g := range grad {
			te.Table[base+i] -= lr * g
		}
	})
	te.gradTable.reset()
}

// ForwardIDs looks up each id in ids and assembles [SeqLen × Dmodel] output.
// Returns utils.ErrVocabOutOfRange if any id is outside [0, VocabSize).
// ids must have length SeqLen; extra IDs are silently ignored, missing IDs
// produce zero-vector rows.
func (te *TokenEmbedding[T]) ForwardIDs(ids []int) ([]T, error) {
	n := min(te.SeqLen, len(ids))
	copy(te.lastIDs, ids[:n])
	for t := range n {
		id := ids[t]
		if id < 0 || id >= te.VocabSize {
			return nil, utils.ErrVocabOutOfRange
		}
		copy(te.lastOut[t*te.Dmodel:(t+1)*te.Dmodel], te.Table[id*te.Dmodel:(id+1)*te.Dmodel])
	}
	return te.lastOut, nil
}

// Forward converts x to integer IDs (truncating to int) and delegates to
// ForwardIDs. This satisfies the layer.Layer[T] interface for topology
// compilers that do not distinguish ID layers.
func (te *TokenEmbedding[T]) Forward(x []T) []T {
	ids := make([]int, len(x))
	for i, v := range x {
		ids[i] = int(v)
	}
	out, _ := te.ForwardIDs(ids)
	return out
}

// Backward accumulates sparse gradients for the token table rows visited during
// the last ForwardIDs call. upstream must be [SeqLen × Dmodel]. Returns nil
// because there is no continuous input to propagate gradients to.
func (te *TokenEmbedding[T]) Backward(upstream []T) []T {
	for t, id := range te.lastIDs {
		te.gradTable.add(id, upstream[t*te.Dmodel:(t+1)*te.Dmodel])
	}
	return nil
}

// tokenEmbeddingJSON is the on-wire representation for MarshalJSON / UnmarshalJSON.
type tokenEmbeddingJSON[T utils.Float] struct {
	Type      string `json:"type"`
	Table     []T    `json:"table"`
	VocabSize int    `json:"vocabSize"`
	SeqLen    int    `json:"seqLen"`
	Dmodel    int    `json:"dmodel"`
}

// MarshalJSON serialises the learnable table; transient caches are omitted.
func (te *TokenEmbedding[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(tokenEmbeddingJSON[T]{
		Type:      "TokenEmbedding",
		VocabSize: te.VocabSize,
		SeqLen:    te.SeqLen,
		Dmodel:    te.Dmodel,
		Table:     te.Table,
	})
}

// UnmarshalJSON restores the table; caches and gradients are lazily re-allocated
// on the first call to Init or ForwardIDs.
func (te *TokenEmbedding[T]) UnmarshalJSON(data []byte) error {
	var j tokenEmbeddingJSON[T]
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	te.VocabSize = j.VocabSize
	te.SeqLen = j.SeqLen
	te.Dmodel = j.Dmodel
	te.Table = j.Table
	te.gradTable = newSparseGrad[T](te.VocabSize, te.Dmodel)
	te.lastIDs = make([]int, te.SeqLen)
	te.lastOut = make([]T, te.SeqLen*te.Dmodel)
	return nil
}
