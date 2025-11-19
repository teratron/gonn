package cell

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

var _ neuron.Nucleus[float32] = (*core[float32])(nil)
var _ neuron.Nucleus[float64] = (*core[float64])(nil)

type core[T utils.Float] struct {
	Id    [2]uint `json:"id" xml:"id"`
	value T
}

func newCore[T utils.Float](id [2]uint) *core[T] {
	return &core[T]{id, 0.0}
}

//func (c *core[T]) GetId() [2]uint {
//	return c.Id
//}

func (c *core[T]) GetValue() *T {
	return &c.value
}

func (c *core[T]) SetValue(value T) {
	c.value = value
}
