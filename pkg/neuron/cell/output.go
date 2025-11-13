package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Neuron[float32] = (*Output[float32])(nil)
var _ neuron.Neuron[float64] = (*Output[float64])(nil)

// Output
type Output[T utils.Float] struct {
	*core[T]
	target *T
}

// NewOutput
func NewOutput[T utils.Float](target *T, activationMode activation.Type, bias bool) *Output[T] {
	return &Output[T]{
		core:   newCore[T](activationMode, bias),
		target: target,
	}
}

// GetTarget
//func (o *Output[T]) GetTarget() *T {
//	return o.target
//}

// SetTarget
func (o *Output[T]) SetTarget(value *T) {
	o.target = value
}

// GetValue
//func (o *Output[T]) GetValue() *T {
//	return &o.value
//}
//
//// GetMiss
//func (o *Output[T]) GetMiss() *T {
//	return &o.miss
//}

// SetMiss
//func (o *Output[T]) SetMiss(miss T) {
//	o.miss = miss
//}

// ----------------------------------------------------------------------------
// FORWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// CalculateValue
func (o *Output[T]) CalculateValue() {
	o.core.CalculateValue()
	o.SetMiss(*o.target - o.value)
}

// ----------------------------------------------------------------------------
// BACKWARD PROPAGATION METHODS
// ----------------------------------------------------------------------------

// CalculateWeight
//func (o *Output[T]) CalculateWeight(rate *T) {
//	_ = o.calculateWeight(rate)
//}

// Forward performs propagation for the output cell
//func (o *Output[T]) Forward() T {
//	value := o.CalculateValue()
//	o.core.value = value
//	return value
//}
//
//// Backward performs propagation of the error
//func (o *Output[T]) Backward(target T) T {
//	o.SetTarget(target)
//
//	// Вычисляем ошибку
//	miss := o.GetMiss()
//	o.core.miss = miss
//
//	// Возвращаем вес для обратного распространения
//	return o.CalculateWeight(miss)
//}
//
//// SetTarget sets the target value
//func (o *Output[T]) SetTarget(target T) {
//	o.target = target
//	o.HasTarget = true
//}
//
//// GetTarget returns the target value
//func (o *Output[T]) GetTarget() T {
//	return o.target
//}
//
//// ClearTarget clears the target value
//func (o *Output[T]) ClearTarget() {
//	o.HasTarget = false
//}
//
//// AddIncomingConnection adds an incoming connection
//func (o *Output[T]) AddIncomingConnection(source nn.Neuron[T], weight T) {
//	o.core.AddIncomingConnection(source, weight)
//}
//
//// GetError returns the current error of the output cell
//func (o *Output[T]) GetError() T {
//	return o.GetMiss()
//}
//
//// IsCorrect checks if the desired accuracy is achieved
//func (o *Output[T]) IsCorrect(tolerance T) bool {
//	if !o.HasTarget {
//		return false
//	}
//	absMiss := o.GetMiss()
//	if absMiss < 0 {
//		absMiss = -absMiss
//	}
//	return absMiss <= tolerance
//}
//
//// Reset resets the state of the output cell
//func (o *Output[T]) Reset() {
//	o.core.miss = 0
//	o.core.value = 0
//	o.HasTarget = false
//}
