package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

type Layer[T utils.Float] struct {
	size       uint
	Activation activation.Type
	Bias       bool
}

func NewLayer[T utils.Float](size uint, activation activation.Type, bias bool) *Layer[T] {
	return &Layer[T]{
		size:       size,
		Activation: activation,
		Bias:       bias,
	}
}
