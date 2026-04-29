package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var (
	_ neuron.Nucleus[float32] = (*Input[float32])(nil)
	_ neuron.Nucleus[float64] = (*Input[float64])(nil)
)

// Input is the cell type that holds an externally supplied feature value.
// Defined as a distinct type (not an alias) over core so the type-tag in
// Id[0] reliably reports neuron.INPUT to network introspection code.
type Input[T utils.Float] core[T]

// NewInput allocates an Input cell pre-populated with value. The position
// index (Id[1]) is set to 0 — layer constructors override it via direct
// field write when assembling the input bundle.
func NewInput[T utils.Float](value T) *Input[T] {
	return &Input[T]{
		Id:    [2]uint{uint(neuron.INPUT), 0},
		value: value,
	}
}

// GetValue mirrors core.GetValue. Declared explicitly because Input is a
// new type, not an alias — methods on core do not promote.
func (i *Input[T]) GetValue() *T {
	return &i.value
}

// SetValue overwrites the stored sample. Used by the dataset feed-loop on
// every training step to swap the input frame without reallocating cells.
func (i *Input[T]) SetValue(value T) {
	i.value = value
}
