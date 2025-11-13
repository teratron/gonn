package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

type Layer[T utils.Float] struct {
	size           uint
	activationMode activation.Type
	Bias           bool
}

func NewLayer[T utils.Float](size uint, activationMode activation.Type, bias bool) *Layer[T] {
	return &Layer[T]{
		size:           size,
		activationMode: activationMode,
		Bias:           bias,
	}
}
