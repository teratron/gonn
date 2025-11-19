package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/utils"
)

type core struct {
	Id   uint `json:"id" xml:"id"`
	Size uint `json:"size" xml:"size"`
}

func newCore[T utils.Float](size uint, activationMode activation.Type, bias bool) *core {
	return &core{}
}
