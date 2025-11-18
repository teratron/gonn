package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Neuron[float32] = (*Dense[float32])(nil)
var _ neuron.Neuron[float64] = (*Dense[float64])(nil)

type Dense[T utils.Float] struct {
	*core[T]
	miss  T
	Axons axon.Bundle[T] `json:"axons" xml:"axons"`
}

func NewDense[T utils.Float](number uint) *Dense[T] {
	return &Dense[T]{
		core:  newCore[T]([2]uint{neuron.DENSE, number}),
		miss:  0.0,
		Axons: make(axon.Bundle[T], 0),
	}
}

func (d *Dense[T]) GetMiss() *T {
	return &d.miss
}

func (d *Dense[T]) SetMiss(value T) {
	d.miss = value
}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION
// ----------------------------------------------------------------------------

func (d *Dense[T]) CalculateValue() {
	d.value = 0.0
	for _, a := range d.Axons {
		d.value += a.CalculateValue()
	}
	//d.value = activation.Activation(d.value, d.activationMode)
	//c.activation.Activation(&c.value)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION
// ----------------------------------------------------------------------------

func (d *Dense[T]) CalculateWeight(rate *T) {
	gradient := *rate * d.miss //* activation.Derivative(d.value, d.activationMode)
	for i := range d.Axons {
		d.Axons[i].CalculateWeight(&gradient)
	}
}
