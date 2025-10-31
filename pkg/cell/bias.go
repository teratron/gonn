package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// BiasCell представляет клетку смещения (bias)
// Содержит только значение T::ONE (в Go это 1.0)
// Аналог Rust BiasCell
type BiasCell[T utils.Float] struct {
	value T // Всегда равно 1.0
}

// NewBiasCell создает новую клетку смещения
func NewBiasCell[T utils.Float]() *BiasCell[T] {
	return &BiasCell[T]{
		value: 1.0,
	}
}

// GetValue возвращает значение клетки смещения (всегда 1.0)
func (b *BiasCell[T]) GetValue() T {
	return b.value
}

// GetMiss возвращает ошибку (всегда 0 для BiasCell, так как bias не обучается)
func (b *BiasCell[T]) GetMiss() T {
	var zero T
	return zero
}

// CalculateValue вычисляет значение (всегда 1.0 для BiasCell)
func (b *BiasCell[T]) CalculateValue() T {
	return b.value
}

// CalculateWeight вычисляет вес (всегда 0 для BiasCell, так как bias не обучается)
func (b *BiasCell[T]) CalculateWeight(miss T) T {
	var zero T
	return zero
}

// Forward выполняет прямое распространение (всегда возвращает 1.0 для BiasCell)
func (b *BiasCell[T]) Forward() T {
	return b.value
}

// Backward выполняет обратное распространение (всегда возвращает 0 для BiasCell, так как bias не обучается)
func (b *BiasCell[T]) Backward(target T) T {
	var zero T
	return zero
}