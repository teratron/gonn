package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*Bias[float32])(nil)
var _ neuron.Nucleus[float64] = (*Bias[float64])(nil)

type _Bias[T utils.Float] _core[T]

func _NewBias[T utils.Float]() *_Bias[T] {
	return &_Bias[T]{
		id:    [2]uint{neuron.BIAS, 0},
		value: 1.0,
	}
}

// Bias
type Bias[T utils.Float] struct {
	value T
}

// NewBias
func NewBias[T utils.Float]() *Bias[T] {
	return &Bias[T]{
		value: 1.0,
	}
}

// GetValue
func (b *Bias[T]) GetValue() *T {
	return &b.value
}
