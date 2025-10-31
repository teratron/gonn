package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// OutputCell представляет выходную клетку нейронной сети
// Содержит CoreCell и target (целевое значение)
// Аналог Rust OutputCell
type OutputCell[T utils.Float] struct {
	Core    *CoreCell[T] // Основная функциональность клетки
	Target  T            // Целевое значение для обучения
	HasTarget bool       // Флаг, указывающий, установлено ли целевое значение
}

// NewOutputCell создает новую выходную клетку
func NewOutputCell[T utils.Float](activationMode ActivationMode) *OutputCell[T] {
	return &OutputCell[T]{
		Core:      NewCoreCell[T](activationMode),
		Target:    0,
		HasTarget: false,
	}
}

// GetValue возвращает значение выходной клетки
func (o *OutputCell[T]) GetValue() T {
	return o.Core.GetValue()
}

// GetMiss возвращает ошибку выходной клетки
func (o *OutputCell[T]) GetMiss() T {
	if o.HasTarget {
		return o.Target - o.GetValue()
	}
	return 0
}

// CalculateValue вычисляет значение выходной клетки
func (o *OutputCell[T]) CalculateValue() T {
	return o.Core.CalculateValue()
}

// CalculateWeight вычисляет вес выходной клетки
func (o *OutputCell[T]) CalculateWeight(miss T) T {
	return o.Core.CalculateWeight(miss)
}

// Forward выполняет прямое распространение для выходной клетки
func (o *OutputCell[T]) Forward() T {
	value := o.CalculateValue()
	o.Core.Value = value
	return value
}

// Backward выполняет обратное распространение ошибки
func (o *OutputCell[T]) Backward(target T) T {
	o.SetTarget(target)
	
	// Вычисляем ошибку
	miss := o.GetMiss()
	o.Core.Miss = miss
	
	// Возвращаем вес для обратного распространения
	return o.CalculateWeight(miss)
}

// SetTarget устанавливает целевое значение
func (o *OutputCell[T]) SetTarget(target T) {
	o.Target = target
	o.HasTarget = true
}

// GetTarget возвращает целевое значение
func (o *OutputCell[T]) GetTarget() T {
	return o.Target
}

// ClearTarget очищает целевое значение
func (o *OutputCell[T]) ClearTarget() {
	o.HasTarget = false
}

// AddIncomingConnection добавляет входящую связь
func (o *OutputCell[T]) AddIncomingConnection(source Neuron[T], weight T) {
	o.Core.AddIncomingConnection(source, weight)
}

// GetError возвращает текущую ошибку выходной клетки
func (o *OutputCell[T]) GetError() T {
	return o.GetMiss()
}

// IsCorrect проверяет, достигнута ли желаемая точность
func (o *OutputCell[T]) IsCorrect(tolerance T) bool {
	if !o.HasTarget {
		return false
	}
	absMiss := o.GetMiss()
	if absMiss < 0 {
		absMiss = -absMiss
	}
	return absMiss <= tolerance
}

// GetSquaredError возвращает квадрат ошибки
func (o *OutputCell[T]) GetSquaredError() T {
	miss := o.GetMiss()
	return miss * miss
}

// Reset сбрасывает состояние выходной клетки
func (o *OutputCell[T]) Reset() {
	o.Core.Reset()
	o.HasTarget = false
}

// SetBias устанавливает значение смещения
func (o *OutputCell[T]) SetBias(bias T) {
	o.Core.SetBias(bias)
}

// GetBias возвращает текущее значение смещения
func (o *OutputCell[T]) GetBias() T {
	return o.Core.GetBias()
}