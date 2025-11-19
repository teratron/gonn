package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

type core struct {
	Id   uint `json:"id" xml:"id"`
	Size uint `json:"size" xml:"size"`
}

func NewLayer[T utils.Float](size uint, activationMode activation.Type, bias bool) *Dense[T] {
	return &Dense[T]{
		core: &core{
			Id:   0,
			Size: size,
		},
		ActivationMode: activationMode,
		Bias:           bias,
	}
}
