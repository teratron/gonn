package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

// core
type core[T utils.Float] struct {
	value          T
	miss           T
	ActivationMode activation.Type
	IncomingAxons  axon.Bundle[T]
}

// newCore
func newCore[T utils.Float]() *core[T] {
	return &core[T]{
		value:          0.0,
		miss:           0.0,
		ActivationMode: activation.SIGMOID,
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
// Forward propagation methods
// ----------------------------------------------------------------------------

// CalculateValue
func (c *core[T]) CalculateValue() {
	c.value = 0.0
	for _, a := range c.IncomingAxons {
		c.value += a.CalculateValue()
	}
	c.value = activation.Activation(c.value, c.ActivationMode)
}

// ----------------------------------------------------------------------------
// Backward propagation methods
// ----------------------------------------------------------------------------

// CalculateWeight
func (c *core[T]) CalculateWeight(rate *T) {
	derivative := activation.Derivative(c.value, c.ActivationMode)
	gradient := *rate * c.miss * derivative
	for i := range c.IncomingAxons {
		c.IncomingAxons[i].CalculateWeight(&gradient)
	}
}
