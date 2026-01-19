package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*Input[float32])(nil)
var _ neuron.Nucleus[float64] = (*Input[float64])(nil)

type Input[T utils.Float] core[T]

func NewInput[T utils.Float](value T) *Input[T] {
	return &Input[T]{
		Id:    [2]uint{uint(neuron.INPUT), 0},
		value: value,
	}
}

func (i *Input[T]) GetValue() *T {
	return &i.value
}

func (i *Input[T]) SetValue(value T) {
	i.value = value
}
