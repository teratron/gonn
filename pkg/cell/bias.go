package cell

import (
	"github.com/teratron/gonn/pkg/utils"
)

// Bias
type Bias[T utils.Float] struct {
	value T
}

// NewBias
func NewBias[T utils.Float]() *Bias[T] {
	return &Bias[T]{
		value: 1.0,
	}
}

// GetValue
func (b *Bias[T]) GetValue() *T {
	return &b.value
}
