package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/axon"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// CoreCell представляет основную функциональность нейронной клетки
// Содержит value, miss, activation_mode, incoming_axons
// Аналог Rust CoreCell
type CoreCell[T utils.Float] struct {
	Value          T               // Текущее значение клетки
	Miss           T               // Ошибка (разница между целевым и полученным значением)
	ActivationMode activation.Type // Тип функции активации
	IncomingAxons  axon.Bundle[T]  // Входящие связи от других клеток
}

// NewCoreCell создает новую клетку с указанным типом активации
func NewCoreCell[T utils.Float](activationMode activation.Type) *CoreCell[T] {
	return &CoreCell[T]{
		Value:          0,
		Miss:           0,
		ActivationMode: activationMode,
		IncomingAxons:  make(axon.Bundle[T], 0),
	}
}

// GetValue возвращает текущее значение клетки
func (c *CoreCell[T]) GetValue() *T {
	return &c.Value
}

// GetMiss возвращает текущую ошибку клетки
func (c *CoreCell[T]) GetMiss() *T {
	return &c.Miss
}

// AddIncomingConnection добавляет входящую связь
func (c *CoreCell[T]) AddIncomingConnection(incoming nn.Nucleus[T], outgoing nn.Neuron[T]) {
	axon := axon.New[T](incoming, outgoing)
	c.IncomingAxons = append(c.IncomingAxons, *axon)
}

// CalculateValue вычисляет значение клетки на основе входящих сигналов (прямое распространение)
func (c *CoreCell[T]) CalculateValue() T {
	c.Value = 0
	for _, a := range c.IncomingAxons {
		c.Value += a.CalculateValue()
	}
	c.Value = activation.Activation(c.Value, c.ActivationMode)
	return c.Value
}

// CalculateWeight вычисляет вес на основе ошибки (обратное распространение)
func (c *CoreCell[T]) CalculateWeight(rate T) T {
	derivative := activation.Derivative(c.Value, c.ActivationMode)
	gradient := rate * c.Miss * derivative
	for i := range c.IncomingAxons {
		c.IncomingAxons[i].CalculateWeight(gradient)
	}
	return gradient
}
