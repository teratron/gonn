package recurrent

import (
	"encoding/json"

	"github.com/teratron/gonn/pkg/layer/conv"
	"github.com/teratron/gonn/pkg/utils"
)

// LastStep collapses a sequence of hidden states [SeqLen*Hidden] to the final
// timestep [Hidden]. It is a stateless, parameter-free reshape that implements
// sequence-to-vector reduction (REC-8).
//
// AI-Meta:
//   - Purpose: Extract the final-timestep hidden vector from a flat recurrent output; no learnable parameters.
//   - Usage: ls := recurrent.NewLastStep[float64](seqLen, hidden); v := ls.Forward(seq).
//   - Concurrency: NotSafe; Forward/Backward store seqLen for the scatter pass.
//   - Related: [GRU], [LSTM], [SimpleRNN], [conv.Layer].
//   - Stability: Stable.
type LastStep[T utils.Float] struct {
	SeqLen int `json:"seq_len"`
	Hidden int `json:"hidden"`
}

// Compile-time assertion: LastStep must satisfy conv.Layer (REC-9).
var (
	_ conv.Layer[float32] = (*LastStep[float32])(nil)
	_ conv.Layer[float64] = (*LastStep[float64])(nil)
)

// NewLastStep constructs a LastStep collapser. No Init call is needed.
//
// AI-Meta:
//   - Purpose: Construct a stateless LastStep collapser for the given sequence shape.
//   - Usage: ls := recurrent.NewLastStep[float64](seqLen, hidden).
//   - Related: [LastStep].
//   - Stability: Stable.
func NewLastStep[T utils.Float](seqLen, hidden int) *LastStep[T] {
	return &LastStep[T]{SeqLen: seqLen, Hidden: hidden}
}

// InputSize returns SeqLen * Hidden (flat sequence input).
func (l *LastStep[T]) InputSize() int { return l.SeqLen * l.Hidden }

// OutputSize returns Hidden (final-timestep vector).
func (l *LastStep[T]) OutputSize() int { return l.Hidden }

// GradSlots returns (nil, nil) — LastStep has no learnable parameters.
func (l *LastStep[T]) GradSlots() (gradW, gradB []T) { return nil, nil }

// Forward picks the final-timestep slice from a flat [SeqLen*Hidden] input.
// Returns a copy of input[(SeqLen-1)*Hidden : SeqLen*Hidden].
//
// AI-Meta:
//   - Purpose: Extract the last-timestep hidden vector from a flat recurrent sequence output.
//   - Concurrency: NotSafe; result slice is owned by this call.
//   - Related: [LastStep.Backward].
//   - Stability: Stable.
func (l *LastStep[T]) Forward(input []T) []T {
	off := (l.SeqLen - 1) * l.Hidden
	out := make([]T, l.Hidden)
	copy(out, input[off:off+l.Hidden])
	return out
}

// Backward scatters the upstream gradient [Hidden] into the final-timestep
// position of a zeroed [SeqLen*Hidden] output. All other timesteps receive zero.
//
// AI-Meta:
//   - Purpose: Scatter upstream gradient to the final-timestep position; zero-fill remaining positions.
//   - Concurrency: NotSafe.
//   - Related: [LastStep.Forward].
//   - Stability: Stable.
func (l *LastStep[T]) Backward(upstream []T) []T {
	out := make([]T, l.SeqLen*l.Hidden)
	off := (l.SeqLen - 1) * l.Hidden
	copy(out[off:off+l.Hidden], upstream)
	return out
}

// MarshalJSON serialises the layer shape for persistence (REC-8).
func (l *LastStep[T]) MarshalJSON() ([]byte, error) {
	type wire struct {
		Type   string `json:"type"`
		SeqLen int    `json:"seq_len"`
		Hidden int    `json:"hidden"`
	}
	return json.Marshal(wire{Type: "LastStep", SeqLen: l.SeqLen, Hidden: l.Hidden})
}

// UnmarshalJSON restores the layer from JSON.
func (l *LastStep[T]) UnmarshalJSON(data []byte) error {
	type wire struct {
		Type   string `json:"type"`
		SeqLen int    `json:"seq_len"`
		Hidden int    `json:"hidden"`
	}
	var w wire
	if err := json.Unmarshal(data, &w); err != nil {
		return utils.Wrap(utils.ErrIntegrity, err, "LastStep.UnmarshalJSON")
	}
	l.SeqLen, l.Hidden = w.SeqLen, w.Hidden
	return nil
}
