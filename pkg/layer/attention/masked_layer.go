package attention

import "github.com/teratron/gonn/pkg/utils"

// MaskedLayer is the optional interface implemented by layers that accept a
// per-call padding mask. The caller sets the mask immediately before Forward;
// the layer consumes it once and clears it so stale masks never leak across
// calls (ATT-6 per-call semantics).
//
// AI-Meta:
//   - Purpose: Optional interface for layers that accept a per-call padding mask (ATT-6).
//   - Usage: if ml, ok := l.(MaskedLayer[T]); ok { ml.SetPaddingMask(mask) }.
//   - Related: [MultiHeadAttention], [layer.Layer].
//   - Stability: Stable.
type MaskedLayer[T utils.Float] interface {
	SetPaddingMask(mask []bool)
}

// compile-time assertion: MultiHeadAttention satisfies MaskedLayer.
var _ MaskedLayer[float32] = (*MultiHeadAttention[float32])(nil)
var _ MaskedLayer[float64] = (*MultiHeadAttention[float64])(nil)

// SetPaddingMask stores mask for the next Forward call. The mask is indexed by
// key/value position: mask[j] = true means position j is valid; false means
// padding (attended weight → 0). Forward calls applyPaddingMask then clears
// padMask to nil so the mask does not persist across calls.
//
// Passing nil disables padding masking for the next call.
func (m *MultiHeadAttention[T]) SetPaddingMask(mask []bool) {
	m.padMask = mask
}
