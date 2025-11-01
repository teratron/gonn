package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/axon"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// Hidden represents a hidden layer of the neural network
// Contains core and outgoing connections (outgoing_axons)
type Hidden[T utils.Float] struct {
	Core          *core[T]
	OutgoingAxons []*axon.Axon[T]
}

// NewHidden creates a new hidden cell
func NewHidden[T utils.Float](activationMode activation.Type) *Hidden[T] {
	return &Hidden[T]{
		Core:          newCore[T](activationMode),
		OutgoingAxons: make([]*axon.Axon[T], 0),
	}
}

// GetValue returns the value of the hidden cell
func (h *Hidden[T]) GetValue() T {
	return h.Core.GetValue()
}

// GetMiss returns the error of the hidden cell
func (h *Hidden[T]) GetMiss() T {
	return h.Core.Miss
}

// CalculateValue calculates the value of the hidden cell
func (h *Hidden[T]) CalculateValue() T {
	return h.Core.CalculateValue()
}

// CalculateWeight calculates the weight of the hidden cell
func (h *Hidden[T]) CalculateWeight(miss T) T {
	return h.Core.CalculateWeight(miss)
}

// Forward performs forward propagation for the hidden cell
func (h *Hidden[T]) Forward() T {
	value := h.CalculateValue()
	h.Core.Value = value
	return value
}

// Backward performs backward propagation of the error
func (h *Hidden[T]) Backward(target T) T {
	// For hidden layers, target is not used directly
	// Error is calculated based on gradients from the next layer
	h.Core.Miss = target
	return h.CalculateWeight(target)
}

// AddOutgoingConnection adds an outgoing connection
func (h *Hidden[T]) AddOutgoingConnection(target nn.Neuron[T], weight T) {
	newAxon := axon.New[T](h.Core, target)
	newAxon.Weight = weight
	h.OutgoingAxons = append(h.OutgoingAxons, newAxon)
}

// PropagateForward propagates the signal forward through all outgoing connections
func (h *Hidden[T]) PropagateForward() {
	h.Forward()
	for _, a := range h.OutgoingAxons {
		_ = a.CalculateValue()
		// Additional logic for accumulating input signals can be added here
	}
}

// PropagateBackward propagates the error backward through all outgoing connections
func (h *Hidden[T]) PropagateBackward(learningRate T) {
	for _, a := range h.OutgoingAxons {
		// Calculate gradient for the connection
		gradient := h.GetMiss() * *a.OutgoingCell.GetValue()
		a.CalculateWeight(gradient * learningRate)
	}
}
