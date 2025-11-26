package layer

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type core[T utils.Float, S neuron.Nucleus[T]] struct {
	Type  uint8 `json:"type" xml:"type"`
	Id    uint  `json:"id" xml:"id"`
	Size  uint  `json:"size" xml:"size"`
	cells []S
}

func newCore[T utils.Float, S neuron.Nucleus[T]](size uint) *core[T, S] {
	return &core[T, S]{
		Type:  neuron.UNKNOWN,
		Id:    0,
		Size:  size,
		cells: make([]S, size),
	}
}
