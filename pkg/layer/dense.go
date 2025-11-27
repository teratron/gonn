package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

type Dense[T utils.Float] struct {
	*base[T, *cell.Dense[T]]
	number  uint
}

func NewDense[T utils.Float](size int, activation activation.Type, bias bool) *Dense[T] {
	d := &Dense[T]{}
	d.Init(size, activation, bias)
	utils.Logger.Info("Dense layer created", "size", size, "activation", activation.String(), "bias", bias)
	return d
}

func (d *Dense[T]) Init(size int, activation activation.Type, bias bool) {
	d.Type = neuron.DENSE
	d.Id = 0
	d.Size = uint(size)
	d.Activation = activation
	d.Bias = bias
	d.cells = make([]*cell.Dense[T], size)
}
