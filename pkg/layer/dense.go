package layer

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Dense[T utils.Float, S neuron.Nucleus[T]] struct {
	*core[T, S]
	Bias           bool            `json:"bias" xml:"bias"`
	ActivationMode activation.Type `json:"activation" xml:"activation"`
}

func (d *Dense[T, S]) Init(size int, activationMode activation.Type, bias bool) {
	d.Size = uint(size)
	d.ActivationMode = activationMode
	d.Bias = bias

	d.core = newCore[T, S](d.Id)
}
