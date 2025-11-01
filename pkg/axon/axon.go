package axon

import (
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// Bundle представляет коллекцию аксонов
type Bundle[T utils.Float] []Axon[T]

// Axon represents a connection between neural network cells
type Axon[T utils.Float] struct {
	// Вес аксона
	Weight T

	// Входная клетка: HiddenCell, InputCell, BiasCell
	IncomingCell nn.Nucleus[T]

	// Выходная клетка: HiddenCell, OutputCell
	OutgoingCell nn.Neuron[T]
}

// New creates a new axon with random weight initialization in range [-0.5, 0.5]
func New[T utils.Float](
	incomingCell nn.Nucleus[T],
	outgoingCell nn.Neuron[T],
) *Axon[T] {
	// Create a local random generator with current time as seed
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	return &Axon[T]{
		Weight:       T(rng.Float64()*1.0 - 0.5), // Случайное значение в диапазоне [-0.5, 0.5]
		IncomingCell: incomingCell,
		OutgoingCell: outgoingCell,
	}
}

// CalculateValue performs forward propagation of the signal
// Formula: *incoming_cell.GetValue() * weight
func (a *Axon[T]) CalculateValue() T {
	return *a.IncomingCell.GetValue() * a.Weight
}

// CalculateMiss performs backward propagation of the error
// Formula: *outgoing_cell.GetMiss() * weight
func (a *Axon[T]) CalculateMiss() T {
	return *a.OutgoingCell.GetMiss() * a.Weight
}

// CalculateWeight updates the axon weight based on gradient
// Formula: weight += gradient * *incoming_cell.GetValue()
func (a *Axon[T]) CalculateWeight(gradient T) {
	a.Weight += gradient * *a.IncomingCell.GetValue()
}
