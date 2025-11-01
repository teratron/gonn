package nn

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Nucleus - base interface for all neural network cells
// Interface for neural network cells with method GetValue()
type Nucleus[T utils.Float] interface {
	// GetValue возвращает текущее значение клетки
	GetValue() *T
}

// Neuron - interface for neurons with learning capability
// Interface for neurons (inherits from Nucleus)
// Contains methods for forward and backward propagation
type Neuron[T utils.Float] interface {
	Nucleus[T]

	// GetMiss returns the error (difference between target and obtained value)
	GetMiss() *T

	// CalculateValue calculates the neuron value based on input signals
	CalculateValue() *T

	// CalculateWeight calculates the neuron weight based on error
	CalculateWeight(*T) T

	// Forward performs forward propagation of the signal
	//Forward() *T

	// Backward performs backward propagation of the error
	//Backward(target *T) *T
}
