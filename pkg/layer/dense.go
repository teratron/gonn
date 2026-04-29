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
// bare struct literal — to close the nil-deref defect from
// [l2-layer-types] §5.3 #1.
type Dense[T utils.Float] struct {
	*base[T, *cell.Dense[T]]
}

// NewDense allocates a Dense layer of the requested size with the chosen
// activation function and bias setting. Cell instances are pre-allocated
// here so callers can wire axons immediately without an explicit Init.
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

// Init reuses the layer with new dimensions. Delegates to base.Init to
// avoid duplicating the activation/bias bookkeeping (closes
// [l2-layer-types] §5.3 #4 — duplicate Init logic between Dense and base).
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

// Cells exposes the dense cell slice. See core.Cells for the ownership
// contract.
func (d *Dense[T]) Cells() []*cell.Dense[T] {
	return d.base.Cells()
}
