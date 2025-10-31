package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// InputCell представляет входную клетку нейронной сети
// Содержит ссылку на T (входные данные)
// Аналог Rust InputCell
type InputCell[T utils.Float] struct {
	value   T     // Текущее значение клетки
	source  *T    // Ссылка на исходные входные данные
	updated bool  // Флаг, указывающий, было ли значение обновлено
}

// NewInputCell создает новую входную клетку
func NewInputCell[T utils.Float](source *T) *InputCell[T] {
	return &InputCell[T]{
		value:   0,
		source:  source,
		updated: false,
	}
}

// GetValue возвращает текущее значение входной клетки
func (i *InputCell[T]) GetValue() T {
	return i.value
}

// SetValue устанавливает значение входной клетки
func (i *InputCell[T]) SetValue(value T) {
	i.value = value
	i.updated = true
}

// UpdateFromSource обновляет значение клетки из исходных данных
func (i *InputCell[T]) UpdateFromSource() {
	if i.source != nil {
		i.value = *i.source
		i.updated = true
	}
}

// GetMiss возвращает ошибку входной клетки (всегда 0, так как входные данные не обучаются)
func (i *InputCell[T]) GetMiss() T {
	return 0
}

// CalculateValue вычисляет значение входной клетки
// Для входной клетки это просто возвращение текущего значения
func (i *InputCell[T]) CalculateValue() T {
	return i.value
}

// CalculateWeight вычисляет вес входной клетки (всегда 0, так как входные данные не обучаются)
func (i *InputCell[T]) CalculateWeight(miss T) T {
	return 0
}

// Forward выполняет прямое распространение для входной клетки
func (i *InputCell[T]) Forward() T {
	// Обновляем значение из источника, если нужно
	if !i.updated && i.source != nil {
		i.UpdateFromSource()
	}
	return i.GetValue()
}

// Backward выполняет обратное распространение ошибки для входной клетки
// Входные клетки не обучаются, поэтому возвращает 0
func (i *InputCell[T]) Backward(target T) T {
	return 0
}

// IsInput возвращает true, так как это входная клетка
func (i *InputCell[T]) IsInput() bool {
	return true
}

// GetSource возвращает ссылку на исходные данные
func (i *InputCell[T]) GetSource() *T {
	return i.source
}

// SetSource устанавливает ссылку на исходные данные
func (i *InputCell[T]) SetSource(source *T) {
	i.source = source
	i.updated = false // Сброс флага обновления
}

// Reset сбрасывает состояние входной клетки
func (i *InputCell[T]) Reset() {
	i.value = 0
	i.updated = false
}

// SetData устанавливает данные напрямую (альтернатива использованию ссылки)
func (i *InputCell[T]) SetData(data T) {
	i.value = data
	i.updated = true
	if i.source != nil {
		*i.source = data // Обновляем также исходные данные
	}
}