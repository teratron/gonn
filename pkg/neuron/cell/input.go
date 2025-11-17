package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*Input[float32])(nil)
var _ neuron.Nucleus[float64] = (*Input[float64])(nil)

//	type _Input[T utils.Float] struct {
//		*_core[T]
//	}
type _Input[T utils.Float] _core[T]

func _NewInput[T utils.Float]() *_Input[T] {
	//return &_Input[T]{
	//	_newCore[T]([2]uint{neuron.INPUT, 0}),
	//}
	return &_Input[T]{
		Id:    [2]uint{neuron.INPUT, 0},
		value: 1.0,
	}
}

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
