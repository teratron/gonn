package nn

import ()

type NN[T float32 | float64] struct {
}

func New[T float32 | float64]() *NN[T] {
	return &NN[T]{}
}
