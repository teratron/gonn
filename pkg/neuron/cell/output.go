package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Neuron[float32] = (*Output[float32])(nil)
var _ neuron.Neuron[float64] = (*Output[float64])(nil)

type Output[T utils.Float] struct {
	*Dense[T]
	target *T
}

func NewOutput[T utils.Float](target *T) *Output[T] {
	return &Output[T]{
		Dense:  NewDense[T](uint(neuron.OUTPUT)),
		target: target,
	}
}

func (o *Output[T]) GetTarget() *T {
	return o.target
}

func (o *Output[T]) SetTarget(value *T) {
	o.target = value
}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION
// ----------------------------------------------------------------------------

func (o *Output[T]) CalculateValue() {
	o.CalculateValue()
	o.SetMiss(*o.target - o.value)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION
// ----------------------------------------------------------------------------
