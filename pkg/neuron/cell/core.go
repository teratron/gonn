package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Neuron[float32] = (*core[float32])(nil)
var _ neuron.Neuron[float64] = (*core[float64])(nil)

type _core[T utils.Float] struct {
	Id    [2]uint `json:"id" xml:"id"`
	value T
}

func _newCore[T utils.Float](id [2]uint) *_core[T] {
	return &_core[T]{id, 0.0}
}

// core
type core[T utils.Float] struct {
	value          T
	miss           T
	activationMode activation.Type
	IncomingAxons  axon.Bundle[T]
	activation     activation.Function[T]
}

// newCore
func newCore[T utils.Float](activationMode activation.Type, bias bool) *core[T] {
	return &core[T]{
		value:          0.0,
		miss:           0.0,
		activationMode: activationMode,
		IncomingAxons:  make(axon.Bundle[T], 0),
	}
}

// GetValue
func (c *core[T]) GetValue() *T {
	return &c.value
}

// SetValue
func (c *core[T]) SetValue(value T) {
	c.value = value
}

// GetMiss
func (c *core[T]) GetMiss() *T {
	return &c.miss
}

// SetMiss
func (c *core[T]) SetMiss(value T) {
	c.miss = value
}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION
// ----------------------------------------------------------------------------

// CalculateValue
func (c *core[T]) CalculateValue() {
	c.value = 0.0
	for _, a := range c.IncomingAxons {
		c.value += a.CalculateValue()
	}
	c.value = activation.Activation(c.value, c.activationMode)
	//c.activation.Activation(&c.value)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION
// ----------------------------------------------------------------------------

// CalculateWeight
func (c *core[T]) CalculateWeight(rate *T) {
	gradient := *rate * c.miss * activation.Derivative(c.value, c.activationMode)
	for i := range c.IncomingAxons {
		c.IncomingAxons[i].CalculateWeight(&gradient)
	}
}
