package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Bias представляет клетку смещения (bias)
type Bias[T utils.Float] struct {
	value T // Всегда равно 1.0
}

// NewBias создает новую клетку смещения
func NewBias[T utils.Float]() *Bias[T] {
	return &Bias[T]{
		value: 1.0,
	}
}

// GetValue возвращает значение клетки смещения (всегда 1.0)
func (b *Bias[T]) GetValue() *T {
	return &b.value
}
