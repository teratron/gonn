package layer

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

type Input[T utils.Float, S neuron.Nucleus[T]] core[T, S]

func NewInput[T utils.Float](size int) *Input[T, *cell.Input[T]] {
	i := &Input[T, *cell.Input[T]]{}
	i.Init(0)
	utils.Logger.Info("Input layer created", "size", 0)
	return i
}

func (i *Input[T, S]) Init(size int) {
	i.Type = neuron.INPUT
	i.Id = 0
	i.Size = uint(size)
	i.cells = make([]S, size)
	utils.Logger.Info("Input layer created", "size", size)
}
