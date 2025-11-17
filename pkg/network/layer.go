package network

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

type layer struct {
	Id   uint `json:"id" xml:"id"`
	Size uint `json:"size" xml:"size"`
}

type Layer[T utils.Float] struct {
	*layer
	ActivationMode activation.Type `json:"activationMode" xml:"activationMode"`
	Bias           bool            `json:"bias" xml:"bias"`
}

type OutputLayer[T utils.Float] struct {
	*Layer[T]
	LossMode loss.Type `json:"lossMode" xml:"lossMode"`
}

func NewLayer[T utils.Float](size uint, activationMode activation.Type, bias bool) *Layer[T] {
	return &Layer[T]{
		layer: &layer{
			Id:   0,
			Size: size,
		},
		ActivationMode: activationMode,
		Bias:           bias,
	}
}
