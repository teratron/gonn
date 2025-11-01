package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Input
type Input[T utils.Float] struct {
	value   T    // Текущее значение клетки
	source  *T   // Ссылка на исходные входные данные
	updated bool // Флаг, указывающий, было ли значение обновлено
}

// NewInput
func NewInput[T utils.Float](source *T) *Input[T] {
	return &Input[T]{
		value:   0,
		source:  source,
		updated: false,
	}
}

// GetValue возвращает текущее значение входной клетки
func (i *Input[T]) GetValue() T {
	return i.value
}

// SetValue устанавливает значение входной клетки
func (i *Input[T]) SetValue(value T) {
	i.value = value
	i.updated = true
}

// UpdateFromSource обновляет значение клетки из исходных данных
func (i *Input[T]) UpdateFromSource() {
	if i.source != nil {
		i.value = *i.source
		i.updated = true
	}
}

// GetMiss возвращает ошибку входной клетки (всегда 0, так как входные данные не обучаются)
func (i *Input[T]) GetMiss() T {
	return 0
}

// CalculateValue вычисляет значение входной клетки
// Для входной клетки это просто возвращение текущего значения
func (i *Input[T]) CalculateValue() T {
	return i.value
}

// CalculateWeight вычисляет вес входной клетки (всегда 0, так как входные данные не обучаются)
func (i *Input[T]) CalculateWeight(miss T) T {
	return 0
}

// Forward выполняет прямое распространение для входной клетки
func (i *Input[T]) Forward() T {
	// Обновляем значение из источника, если нужно
	if !i.updated && i.source != nil {
		i.UpdateFromSource()
	}
	return i.GetValue()
}

// Backward выполняет обратное распространение ошибки для входной клетки
// Входные клетки не обучаются, поэтому возвращает 0
func (i *Input[T]) Backward(target T) T {
	return 0
}

// IsInput возвращает true, так как это входная клетка
func (i *Input[T]) IsInput() bool {
	return true
}

// GetSource возвращает ссылку на исходные данные
func (i *Input[T]) GetSource() *T {
	return i.source
}

// SetSource устанавливает ссылку на исходные данные
func (i *Input[T]) SetSource(source *T) {
	i.source = source
	i.updated = false // Сброс флага обновления
}

// Reset сбрасывает состояние входной клетки
func (i *Input[T]) Reset() {
	i.value = 0
	i.updated = false
}

// SetData устанавливает данные напрямую (альтернатива использованию ссылки)
func (i *Input[T]) SetData(data T) {
	i.value = data
	i.updated = true
	if i.source != nil {
		*i.source = data // Обновляем также исходные данные
	}
}
