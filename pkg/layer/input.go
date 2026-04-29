package layer

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Input is the entry-point layer type. Aliased to the unexported core
// container parameterised by *cell.Input[T] — Input cells only carry a
// scalar value, no axons or activation, so the lighter core (no base
// embedding) is sufficient.
type Input[T utils.Float] core[T, *cell.Input[T]]

// NewInput allocates an Input layer of the requested size and populates
// it with zero-valued cell.Input units. Closes [l2-layer-types] §5.3 #3:
// the previous implementation passed a hard-coded 0 to Init and silently
// produced a layer of size zero regardless of the caller's request.
func NewInput[T utils.Float](size int) *Input[T] {
	in := (*Input[T])(newCore[T, *cell.Input[T]](neuron.INPUT, size))
	in.populate()
	utils.Logger.Info("Input layer created", "size", size)
	return in
}

// Init resets the Input layer to the requested size and re-populates
// cells. Honours the call-supplied size — no implicit zeroing.
func (i *Input[T]) Init(size int) {
	((*core[T, *cell.Input[T]])(i)).Init(neuron.INPUT, size)
	i.populate()
}

// populate fills the cell slice with freshly allocated cell.Input units.
// Each cell holds a zero value initially — callers feed real samples via
// SetValue at runtime.
func (i *Input[T]) populate() {
	c := (*core[T, *cell.Input[T]])(i)
	for idx := range c.cells {
		c.cells[idx] = cell.NewInput[T](0)
	}
}

// Cells exposes the input cell slice for read access. See core.Cells for
// the ownership contract.
func (i *Input[T]) Cells() []*cell.Input[T] {
	return ((*core[T, *cell.Input[T]])(i)).Cells()
}
