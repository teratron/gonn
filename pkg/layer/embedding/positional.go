package embedding

import (
	"encoding/json"
	"math/rand/v2"

	"github.com/teratron/gonn/pkg/layer"
	"github.com/teratron/gonn/pkg/utils"
)

// PositionalMode selects how position information is encoded.
type PositionalMode uint8

const (
	// Sinusoidal uses the fixed Vaswani et al. 2017 encoding (no learned params).
	Sinusoidal PositionalMode = iota
	// Learnable uses a trainable position embedding table initialised with
	// XavierUniform values.
	Learnable
)

// PositionalEncoding adds position information to an input embedding matrix.
// It covers both fixed sinusoidal and learnable variants via the Mode field.
// Input / output shape: [SeqLen × Dmodel] flat row-major.
//
// AI-Meta:
//   - Purpose: Sinusoidal or learnable positional encoding implementing EMB-5..EMB-8.
//   - Usage: Compose with TokenEmbedding inside EmbeddingStack or via nn.WithPositionalEncoding.
//   - Related: [TokenEmbedding], [EmbeddingStack], [buildSinusoidalTable].
//   - Stability: Stable.
type PositionalEncoding[T utils.Float] struct {
	Table     []T `json:"table"`
	gradTable []T
	SeqLen    int            `json:"seqLen"`
	Dmodel    int            `json:"dmodel"`
	Mode      PositionalMode `json:"mode"`
}

// compile-time assertions.
var _ layer.Layer[float32] = (*PositionalEncoding[float32])(nil)
var _ layer.Layer[float64] = (*PositionalEncoding[float64])(nil)

// NewPositionalEncoding creates an uninitialised PositionalEncoding. Call Init before use.
func NewPositionalEncoding[T utils.Float](seqLen, dmodel int, mode PositionalMode) *PositionalEncoding[T] {
	if seqLen < 1 {
		seqLen = 1
	}
	if dmodel < 1 {
		dmodel = 1
	}
	return &PositionalEncoding[T]{SeqLen: seqLen, Dmodel: dmodel, Mode: mode}
}

// Init builds the positional table and pre-allocates caches.
// For Sinusoidal mode the table is deterministic; rng is ignored.
// For Learnable mode the table is initialised with XavierUniform.
func (pe *PositionalEncoding[T]) Init(rng *rand.Rand) {
	sz := pe.SeqLen * pe.Dmodel
	switch pe.Mode {
	case Sinusoidal:
		pe.Table = buildSinusoidalTable[T](pe.SeqLen, pe.Dmodel)
		pe.gradTable = nil
	case Learnable:
		pe.Table = make([]T, sz)
		for i := range pe.Table {
			pe.Table[i] = utils.XavierUniform[T](rng, pe.SeqLen, pe.Dmodel)
		}
		pe.gradTable = make([]T, sz)
	}
}

// InputSize returns SeqLen*Dmodel.
func (pe *PositionalEncoding[T]) InputSize() int { return pe.SeqLen * pe.Dmodel }

// OutputSize returns SeqLen*Dmodel.
func (pe *PositionalEncoding[T]) OutputSize() int { return pe.SeqLen * pe.Dmodel }

// GradSlots returns (gradTable, nil) for Learnable mode; (nil, nil) for Sinusoidal.
func (pe *PositionalEncoding[T]) GradSlots() (gradW, gradB []T) {
	return pe.gradTable, nil
}

// ApplyGradSGD updates the learnable positional table in place, then zeroes the
// gradient accumulator. Sinusoidal mode is a no-op (fixed table). Resetting the
// accumulator here fixes the unbounded accumulation bug (audit C4) where the
// gradient grew every step because it was applied but never cleared.
func (pe *PositionalEncoding[T]) ApplyGradSGD(lr T) {
	if pe.Mode != Learnable || pe.gradTable == nil {
		return
	}
	n := min(len(pe.Table), len(pe.gradTable))
	for i := range n {
		pe.Table[i] -= lr * pe.gradTable[i]
	}
	clear(pe.gradTable)
}

// Forward adds the positional table to x element-wise. x must be [SeqLen×Dmodel].
// The result is written into a fresh slice. (No input cache is kept — the
// additive Backward never needs x; the historical lastIn buffer was dead.)
func (pe *PositionalEncoding[T]) Forward(x []T) []T {
	sz := pe.SeqLen * pe.Dmodel
	out := make([]T, sz)
	for i := 0; i < sz && i < len(x); i++ {
		out[i] = x[i]
		// Table is nil during the compile-time shape-resolution walk (before
		// Init); treat the positional offset as zero there.
		if i < len(pe.Table) {
			out[i] += pe.Table[i]
		}
	}
	return out
}

// Backward receives upstream gradient [SeqLen×Dmodel].
// In Sinusoidal mode the positional table is constant, so the gradient passes
// through unchanged and no table gradient is accumulated.
// In Learnable mode the gradient is also accumulated into gradTable.
func (pe *PositionalEncoding[T]) Backward(upstream []T) []T {
	if pe.Mode == Learnable {
		for i, v := range upstream {
			pe.gradTable[i] += v
		}
	}
	// gradient wrt input is upstream (addition is its own derivative)
	dIn := make([]T, len(upstream))
	copy(dIn, upstream)
	return dIn
}

// positionalEncodingJSON is the on-wire representation.
type positionalEncodingJSON[T utils.Float] struct {
	Type   string         `json:"type"`
	Table  []T            `json:"table"`
	SeqLen int            `json:"seqLen"`
	Dmodel int            `json:"dmodel"`
	Mode   PositionalMode `json:"mode"`
}

// MarshalJSON serialises the table and mode; caches are omitted.
func (pe *PositionalEncoding[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(positionalEncodingJSON[T]{
		Type:   "PositionalEncoding",
		SeqLen: pe.SeqLen,
		Dmodel: pe.Dmodel,
		Mode:   pe.Mode,
		Table:  pe.Table,
	})
}

// UnmarshalJSON restores mode and table; caches are re-allocated.
func (pe *PositionalEncoding[T]) UnmarshalJSON(data []byte) error {
	var j positionalEncodingJSON[T]
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	pe.SeqLen = j.SeqLen
	pe.Dmodel = j.Dmodel
	pe.Mode = j.Mode
	pe.Table = j.Table
	sz := pe.SeqLen * pe.Dmodel
	if pe.Mode == Learnable {
		pe.gradTable = make([]T, sz)
	}
	return nil
}
