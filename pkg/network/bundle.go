package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

type Bundle[T utils.Float, S neuron.Nucleus[T]] struct {
	Cells  []S
	number uint
}

func NewBundle[T utils.Float, S neuron.Nucleus[T]](/*data *[]S*/) *Bundle[T, S] {
	//number := uint(len(*data))
	return &Bundle[T, S]{
		Cells:  make([]S, 0),
		number: 0,
	}
}
