package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/axon"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// core represents the core functionality of a neural cell
// Contains value, miss, activation_mode, incoming_axons
type core[T utils.Float] struct {
	Value          T               // Текущее значение клетки
	Miss           T               // Ошибка (разница между целевым и полученным значением)
	ActivationMode activation.Type // Тип функции активации
	IncomingAxons  axon.Bundle[T]  // Входящие связи от других клеток
}

// newCore creates a new cell with specified activation type
func newCore[T utils.Float](activationMode activation.Type) *core[T] {
	return &core[T]{
		Value:          0,
		Miss:           0,
		ActivationMode: activationMode,
		IncomingAxons:  make(axon.Bundle[T], 0),
	}
}

// GetValue returns the current cell value
func (c *core[T]) GetValue() *T {
	return &c.Value
}

// GetMiss returns the current cell error
func (c *core[T]) GetMiss() *T {
	return &c.Miss
}

// AddIncomingConnection adds an incoming connection
func (c *core[T]) AddIncomingConnection(incoming nn.Nucleus[T], outgoing nn.Neuron[T]) {
	c.IncomingAxons = append(c.IncomingAxons, *axon.New[T](incoming, outgoing))
}

// CalculateValue calculates the cell value based on incoming signals (forward propagation)
func (c *core[T]) CalculateValue() T {
	c.Value = 0
	for _, a := range c.IncomingAxons {
		c.Value += a.CalculateValue()
	}
	c.Value = activation.Activation(c.Value, c.ActivationMode)
	return c.Value
}

// CalculateWeight calculates weight based on error (backward propagation)
func (c *core[T]) CalculateWeight(rate T) T {
	derivative := activation.Derivative(c.Value, c.ActivationMode)
	gradient := rate * c.Miss * derivative
	for i := range c.IncomingAxons {
		c.IncomingAxons[i].CalculateWeight(gradient)
	}
	return gradient
}
