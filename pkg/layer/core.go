// Package layer — neural network layer container types.
//
// Implements [l2-layer-types] §5: a three-tier composition chain
// `core → base → Input/Dense/Output`. core owns the shared identity
// (Type, Id, Size) and the cell slice; base extends it with activation
// and bias; concrete types specialise per role.
package layer

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// core is the unexported foundation shared by every layer type. It holds
// the layer kind tag, the position index inside the network, the declared
// size, and the cell slice itself. Every concrete constructor is required
// to allocate core via newCore so the embedded pointer is non-nil — this
// closes the nil-deref defect catalogued in [l2-layer-types] §5.3 #1-2.
type core[T utils.Float, S neuron.Nucleus[T]] struct {
	cells []S
	Id    uint  `json:"id" xml:"id"`
	Size  uint  `json:"size" xml:"size"`
	Type  uint8 `json:"type" xml:"type"`
}

// newCore allocates a core ready for cell population. The cells slice is
// allocated with the declared length (zero-valued elements); the concrete
// layer type fills it via populateCells / its Init method. size is taken
// as int because layer constructors accept negative or zero values from
// user code and translate them to a size-zero layer (still valid in
// dynamic topology mode — see INV-2).
func newCore[T utils.Float, S neuron.Nucleus[T]](kind uint8, size int) *core[T, S] {
	if size < 0 {
		size = 0
	}
	return &core[T, S]{
		Type:  kind,
		Id:    0,
		Size:  uint(size),
		cells: make([]S, size),
	}
}

// Init resets a core in place to the supplied dimensions. Used by Layer
// types that re-initialise an existing pointer (Dense.Init delegates here
// rather than re-implementing the body — closes the duplicate-Init defect
// from [l2-layer-types] §5.3 #4).
func (c *core[T, S]) Init(kind uint8, size int) {
	if size < 0 {
		size = 0
	}
	c.Type = kind
	c.Size = uint(size)
	c.cells = make([]S, size)
}

// Cells exposes the raw cell slice. Callers must not append or reslice —
// the layer owns the backing array. Returned for read-only iteration only.
func (c *core[T, S]) Cells() []S {
	return c.cells
}

// SetCell installs a concrete cell at position idx. Panics if idx is out
// of range: a programming error in the layer constructor, never a runtime
// condition.
func (c *core[T, S]) SetCell(idx int, s S) {
	c.cells[idx] = s
}
