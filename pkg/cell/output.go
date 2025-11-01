package cell

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// Output represents an output cell of the neural network
// Contains core and target (desired value)
type Output[T utils.Float] struct {
	Core      *core[T] // Основная функциональность клетки
	Target    T        // Целевое значение для обучения
	HasTarget bool     // Флаг, указывающий, установлено ли целевое значение
}

// NewOutput creates a new output cell
func NewOutput[T utils.Float](activationMode activation.Type) *Output[T] {
	return &Output[T]{
		Core:      newCore[T](activationMode),
		Target:    0,
		HasTarget: false,
	}
}

// GetValue returns the value of the output cell
func (o *Output[T]) GetValue() T {
	return o.Core.GetValue()
}

// GetMiss returns the error of the output cell
func (o *Output[T]) GetMiss() T {
	if o.HasTarget {
		return o.Target - o.GetValue()
	}
	return 0
}

// CalculateValue calculates the value of the output cell
func (o *Output[T]) CalculateValue() T {
	return o.Core.CalculateValue()
}

// CalculateWeight calculates the weight of the output cell
func (o *Output[T]) CalculateWeight(miss T) T {
	return o.Core.CalculateWeight(miss)
}

// Forward performs forward propagation for the output cell
func (o *Output[T]) Forward() T {
	value := o.CalculateValue()
	o.Core.Value = value
	return value
}

// Backward performs backward propagation of the error
func (o *Output[T]) Backward(target T) T {
	o.SetTarget(target)

	// Вычисляем ошибку
	miss := o.GetMiss()
	o.Core.Miss = miss

	// Возвращаем вес для обратного распространения
	return o.CalculateWeight(miss)
}

// SetTarget sets the target value
func (o *Output[T]) SetTarget(target T) {
	o.Target = target
	o.HasTarget = true
}

// GetTarget returns the target value
func (o *Output[T]) GetTarget() T {
	return o.Target
}

// ClearTarget clears the target value
func (o *Output[T]) ClearTarget() {
	o.HasTarget = false
}

// AddIncomingConnection adds an incoming connection
func (o *Output[T]) AddIncomingConnection(source nn.Neuron[T], weight T) {
	o.Core.AddIncomingConnection(source, weight)
}

// GetError returns the current error of the output cell
func (o *Output[T]) GetError() T {
	return o.GetMiss()
}

// IsCorrect checks if the desired accuracy is achieved
func (o *Output[T]) IsCorrect(tolerance T) bool {
	if !o.HasTarget {
		return false
	}
	absMiss := o.GetMiss()
	if absMiss < 0 {
		absMiss = -absMiss
	}
	return absMiss <= tolerance
}

// GetSquaredError returns the squared error
func (o *Output[T]) GetSquaredError() T {
	miss := o.GetMiss()
	return miss * miss
}

// Reset resets the state of the output cell
func (o *Output[T]) Reset() {
	o.Core.Miss = 0
	o.Core.Value = 0
	o.HasTarget = false
}

// SetBias sets the bias value
func (o *Output[T]) SetBias(bias T) {
	// core[T] не имеет поля Bias, поэтому просто устанавливаем значение
	o.Core.Value = bias
}

// GetBias returns the current bias value
func (o *Output[T]) GetBias() T {
	// core[T] не имеет поля Bias, возвращаем текущее значение
	return o.Core.Value
}
