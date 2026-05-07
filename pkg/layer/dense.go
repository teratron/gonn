package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Dense is the canonical hidden layer type. Embeds *base[T, *cell.Dense[T]]
// so it inherits Activation, Bias, and the core identity fields. The
// embedded pointer is allocated via newBase in the constructor — never a
// bare struct literal.
//
// AI-Meta:
//   - Purpose: Hidden layer carrying fully connected Dense cells with activation and optional bias.
//   - Concurrency: NotSafe; CalculateValue and CalculateWeight mutate cell state.
//   - Related: [NewDense], [cell.Dense], [Input], [Output].
type Dense[T utils.Float] struct {
	*base[T, *cell.Dense[T]]
}

// NewDense allocates a Dense layer of the requested size with the chosen
// activation function and bias setting. Cell instances are pre-allocated
// here so callers can wire axons immediately without an explicit Init.
//
// AI-Meta:
//   - Purpose: Construct a Dense layer with pre-allocated cells; used by the network builder.
//   - Usage: l := layer.NewDense[float32](8, activation.ReLU, true).
//   - Related: [Dense], [Init], [cell.NewDense].
func NewDense[T utils.Float](size int, act activation.Type, useBias bool) *Dense[T] {
	d := &Dense[T]{
		base: newBase[T, *cell.Dense[T]](neuron.DENSE, size, act, useBias),
	}
	d.populate()
	utils.Logger.Info("Dense layer created",
		"size", size,
		"activation", act.String(),
		"bias", useBias,
	)
	return d
}

// Init reuses the layer with new dimensions, delegating activation/bias
// bookkeeping to base.Init.
//
// AI-Meta:
//   - Purpose: Resize and re-initialise the Dense layer; reuses existing base or allocates a new one.
//   - Concurrency: NotSafe; must not run during a forward or backward pass.
//   - Related: [NewDense].
func (d *Dense[T]) Init(size int, act activation.Type, useBias bool) {
	if d.base == nil {
		d.base = newBase[T, *cell.Dense[T]](neuron.DENSE, size, act, useBias)
	} else {
		d.base.Init(neuron.DENSE, size, act, useBias)
	}
	d.populate()
}

// populate fills the slice with cell.Dense units indexed by their slot
// position. The position index is the second component of the cell's Id
// tuple — Track D's network builder uses it to disambiguate cells when
// the network owns more than one Dense layer.
func (d *Dense[T]) populate() {
	for idx := range d.cells {
		d.cells[idx] = cell.NewDense[T](uint(idx))
	}
}

// Cells exposes the dense cell slice for axon wiring and backprop access.
//
// AI-Meta:
//   - Purpose: Return the cell slice so the network builder can wire axons and run passes.
//   - Concurrency: ReadSafe; elements are mutated only by CalculateValue/CalculateWeight.
//   - Related: [cell.Dense].
func (d *Dense[T]) Cells() []*cell.Dense[T] {
	return d.base.Cells()
}
