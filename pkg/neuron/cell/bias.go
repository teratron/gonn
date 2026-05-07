package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var (
	_ neuron.Nucleus[float32] = (*Bias[float32])(nil)
	_ neuron.Nucleus[float64] = (*Bias[float64])(nil)
)

// Bias is a fixed-output cell that always reports 1.0 — the canonical
// bias unit attached to dense layers so the network can learn an additive
// offset alongside the input-driven weights.
//
// AI-Meta:
//   - Purpose: Constant-output cell providing an always-on 1.0 bias signal to dense layers.
//   - Usage: Created automatically by newBase when useBias=true; accessed via layer.BiasCell().
//   - Related: [NewBias], [neuron.Nucleus].
type Bias[T utils.Float] core[T]

// NewBias allocates a Bias cell with value pinned to 1.0.
//
// AI-Meta:
//   - Purpose: Construct a Bias cell; used internally by layer constructors when bias is enabled.
//   - Usage: b := cell.NewBias[float32]().
//   - Related: [Bias].
func NewBias[T utils.Float]() *Bias[T] {
	return &Bias[T]{
		Id:    [2]uint{uint(neuron.BIAS), 0},
		value: 1.0,
	}
}

// GetValue returns a pointer to the constant 1.0 scalar. Bias never
// exposes SetValue — the value is invariant by definition.
func (b *Bias[T]) GetValue() *T {
	return &b.value
}
