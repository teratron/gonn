package layer

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Input[T utils.Float, S neuron.Nucleus[T]] core[T, S]

func NewInput[T utils.Float, S neuron.Nucleus[T]]() *Input[T, S] {
	return &Input[T, S]{}
}
