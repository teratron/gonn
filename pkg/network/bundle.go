package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// bundle is the typed cell container owned by Network. Each Network field
// (Input, Hidden, Output) holds a bundle parameterised by the matching
// concrete cell type. The container intentionally exposes very little
// surface — Network methods do the type-assertion per element when they
// need to reach the [neuron.Neuron] interface (slice covariance is not
// available in Go, so any whole-slice cast was unsound and silently
// no-op'd in the legacy code; closes [l2-network-graph] §5.4 #3).
type bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	cells []S
}

func newBundle[T utils.Float, S neuron.Nucleus[T]]() bundle[T, S] {
	return bundle[T, S]{cells: make([]S, 0)}
}

// Replace swaps the cell slice for the supplied one — used by SetLayers
// to install the cells produced by a layer constructor.
func (b *bundle[T, S]) Replace(cells []S) {
	b.cells = cells
}

// Add appends a single cell to the bundle.
func (b *bundle[T, S]) Add(c S) {
	b.cells = append(b.cells, c)
}

// Len returns the number of cells in the bundle.
func (b *bundle[T, S]) Len() int {
	return len(b.cells)
}

// Cells exposes the slice for read-only iteration. Callers must not
// append, reslice, or mutate elements via the returned reference — the
// Network owns the storage.
func (b *bundle[T, S]) Cells() []S {
	return b.cells
}

// At returns the cell at index idx. Panics on out-of-range — programming
// error in the caller, never a runtime condition.
func (b *bundle[T, S]) At(idx int) S {
	return b.cells[idx]
}
