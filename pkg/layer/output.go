package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

type Output[T utils.Float, S neuron.Neuron[T]] struct {
	*Dense[T, S]
	Loss loss.Type `json:"loss" xml:"loss"`
}

func NewOutput[T utils.Float](size int, activation activation.Type, loss loss.Type, bias bool) *Output[T, *cell.Output[T]] {
	o := &Output[T, *cell.Output[T]]{}
	o.Init(size, activation, loss, bias)
	utils.Logger.Info("Output layer created", "size", size, "activation", activation.String(), "loss", loss.String(), "bias", bias)
	return o
}

func (o *Output[T, S]) Init(size int, activation activation.Type, loss loss.Type, bias bool) {
	o.Type = neuron.OUTPUT
	o.Id = 0
	o.Dense.Init(size, activation, bias)
	o.Loss = loss
}
