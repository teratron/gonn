package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type base[T utils.Float, S neuron.Neuron[T]] struct {
	*core[T, S]
	Bias       bool            `json:"bias" xml:"bias"`
	Activation activation.Type `json:"activation" xml:"activation"`
}

func newBase[T utils.Float, S neuron.Neuron[T]](size int, activation activation.Type, bias bool) *base[T, S] {
	d := &base[T, S]{}
	d.Init(size, activation, bias)
	return d
}

func (d *base[T, S]) Init(size int, activation activation.Type, bias bool) {
	d.Type = neuron.DENSE
	d.Id = 0
	d.Size = uint(size)
	d.Activation = activation
	d.Bias = bias
	d.cells = make([]S, size)
}
