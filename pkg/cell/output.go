package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Output
type Output[T utils.Float] struct {
	*core[T]
	Target *T
}

// NewOutput
func NewOutput[T utils.Float](target *T) *Output[T] {
	return &Output[T]{
		core:   newCore[T](),
		Target: target,
	}
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

// CalculateValue
func (o *Output[T]) CalculateValue() {
	o.core.CalculateValue()
	o.SetMiss(*o.Target - o.value)
}

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
//	o.Target = target
//	o.HasTarget = true
//}
//
//// GetTarget returns the target value
//func (o *Output[T]) GetTarget() T {
//	return o.Target
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
//// GetSquaredError returns the squared error
//func (o *Output[T]) GetSquaredError() T {
//	miss := o.GetMiss()
//	return miss * miss
//}
//
//// Reset resets the state of the output cell
//func (o *Output[T]) Reset() {
//	o.core.miss = 0
//	o.core.value = 0
//	o.HasTarget = false
//}
//
//// SetBias sets the bias value
//func (o *Output[T]) SetBias(bias T) {
//	// core[T] не имеет поля Bias, поэтому просто устанавливаем значение
//	o.core.value = bias
//}
//
//// GetBias returns the current bias value
//func (o *Output[T]) GetBias() T {
//	// core[T] не имеет поля Bias, возвращаем текущее значение
//	return o.core.value
//}
