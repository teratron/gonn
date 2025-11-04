package axon

import (
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// Bundle представляет коллекцию аксонов
type Bundle[T utils.Float] []Axon[T]

// Axon represents a connection between neural network cells
type Axon[T utils.Float] struct {
	// Вес аксона
	Weight T

	// Входная клетка: Hidden, Input, Bias
	IncomingCell neuron.Nucleus[T]

	// Выходная клетка: Hidden, Output
	OutgoingCell neuron.Neuron[T]
}

// New creates a new axon with random weight initialization in range [-0.5, 0.5]
func New[T utils.Float](
	incomingCell neuron.Nucleus[T],
	outgoingCell neuron.Neuron[T],
) *Axon[T] {
	// Create a local random generator with current time as seed
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return &Axon[T]{
		Weight:       T(rng.Float64()*1.0 - 0.5), // Случайное значение в диапазоне [-0.5, 0.5]
		IncomingCell: incomingCell,
		OutgoingCell: outgoingCell,
	}
}

// ----------------------------------------------------------------------------
// Forward propagation methods
// ----------------------------------------------------------------------------

// CalculateValue
func (a *Axon[T]) CalculateValue() T {
	return *a.IncomingCell.GetValue() * a.Weight
}

// ----------------------------------------------------------------------------
// Backward propagation methods
// ----------------------------------------------------------------------------

// CalculateMiss
func (a *Axon[T]) CalculateMiss() T {
	return *a.OutgoingCell.GetMiss() * a.Weight
}

// CalculateWeight
func (a *Axon[T]) CalculateWeight(gradient *T) {
	a.Weight += *gradient * *a.IncomingCell.GetValue()
}
