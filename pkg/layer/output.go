package layer

import (
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Output[T utils.Float, S neuron.Nucleus[T]] struct {
	*Dense[T, S]
	LossMode loss.Type `json:"loss" xml:"loss"`
}
