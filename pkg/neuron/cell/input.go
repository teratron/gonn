package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*Input[float32])(nil)
var _ neuron.Nucleus[float64] = (*Input[float64])(nil)

// Input
type Input[T utils.Float] struct {
	value *T
}

// NewInput
func NewInput[T utils.Float](value *T) *Input[T] {
	return &Input[T]{
		value,
	}
}

// GetValue
func (i *Input[T]) GetValue() *T {
	return i.value
}

// SetValue
func (i *Input[T]) SetValue(value *T) {
	i.value = value
}
