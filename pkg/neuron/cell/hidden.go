package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Neuron[float32] = (*Hidden[float32])(nil)
var _ neuron.Neuron[float64] = (*Hidden[float64])(nil)

// Hidden
type Hidden[T utils.Float] struct {
	*core[T]
	OutgoingAxons axon.Bundle[T]
}

// NewHidden
func NewHidden[T utils.Float]() *Hidden[T] {
	return &Hidden[T]{
		core:          newCore[T](),
		OutgoingAxons: make(axon.Bundle[T], 0),
	}
}

// GetValue
//func (h *Hidden[T]) GetValue() *T {
//	return &h.value
//}
//
//// GetMiss
//func (h *Hidden[T]) GetMiss() *T {
//	return &h.miss
//}

// SetMiss
//func (h *Hidden[T]) SetMiss(miss T) {
//	h.miss = miss
//}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// CalculateValue
//func (h *Hidden[T]) CalculateValue() {
//	h.calculateValue()
//}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// CalculateWeight
//func (h *Hidden[T]) CalculateWeight(rate *T) {
//	h.calculateWeight(rate)
//}

// Forward performs propagation for the hidden cell
//func (h *Hidden[T]) Forward() T {
//	value := h.CalculateValue()
//	h.core.value = value
//	return value
//}
//
//// Backward performs propagation of the error
//func (h *Hidden[T]) Backward(target T) T {
//	// For hidden layers, target is not used directly
//	// Error is calculated based on gradients from the next layer
//	h.core.miss = target
//	return h.CalculateWeight(target)
//}

// AddOutgoingConnection adds an outgoing connection
//func (h *Hidden[T]) AddOutgoingConnection(target nn.Neuron[T], weight T) {
//	newAxon := axon.New[T](h.core, target)
//	newAxon.Weight = weight
//	h.OutgoingAxons = append(h.OutgoingAxons, newAxon)
//}
//
//// PropagateForward propagates the signal forward through all outgoing connections
//func (h *Hidden[T]) PropagateForward() {
//	h.Forward()
//	for _, a := range h.OutgoingAxons {
//		_ = a.CalculateValue()
//		// Additional logic for accumulating input signals can be added here
//	}
//}
//
//// PropagateBackward propagates the error backward through all outgoing connections
//func (h *Hidden[T]) PropagateBackward(learningRate T) {
//	for _, a := range h.OutgoingAxons {
//		// Calculate gradient for the connection
//		gradient := h.GetMiss() * *a.OutgoingCell.GetValue()
//		a.CalculateWeight(gradient * learningRate)
//	}
//}
