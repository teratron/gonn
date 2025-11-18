package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*Bias[float32])(nil)
var _ neuron.Nucleus[float64] = (*Bias[float64])(nil)

type Bias[T utils.Float] core[T]

func _NewBias[T utils.Float]() *Bias[T] {
	return &Bias[T]{
		Id:    [2]uint{neuron.BIAS, 0},
		value: 1.0,
	}
}

func (b *Bias[T]) GetValue() *T {
	return &b.value
}
