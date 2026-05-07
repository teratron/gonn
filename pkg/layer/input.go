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
//
// AI-Meta:
//   - Purpose: First layer in the network graph; holds one Input cell per feature.
//   - Concurrency: NotSafe; cells are mutated via SetValue before each forward pass.
//   - Related: [NewInput], [cell.Input], [Dense], [Output].
type Input[T utils.Float] core[T, *cell.Input[T]]

// NewInput allocates an Input layer of the requested size and populates
// it with zero-valued cell.Input units.
//
// AI-Meta:
//   - Purpose: Construct an Input layer pre-filled with zero-valued cells; used by the network builder.
//   - Usage: l := layer.NewInput[float32](4) for a four-feature input layer.
//   - Related: [Input], [Init].
func NewInput[T utils.Float](size int) *Input[T] {
	in := (*Input[T])(newCore[T, *cell.Input[T]](neuron.INPUT, size))
	in.populate()
	utils.Logger.Info("Input layer created", "size", size)
	return in
}

// Init resets the Input layer to the requested size and re-populates cells.
//
// AI-Meta:
//   - Purpose: Resize and re-initialise the Input layer, discarding previous cells.
//   - Concurrency: NotSafe; must not be called concurrently with forward passes.
//   - Related: [NewInput].
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

// Cells exposes the input cell slice for read access.
//
// AI-Meta:
//   - Purpose: Return the cell slice so callers can feed feature values via cell.SetValue.
//   - Concurrency: ReadSafe; slice elements must be written only before the forward pass.
//   - Related: [cell.Input].
func (i *Input[T]) Cells() []*cell.Input[T] {
	return ((*core[T, *cell.Input[T]])(i)).Cells()
}
