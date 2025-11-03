package network

import (
	"github.com/teratron/gonn/pkg/cell"
	"github.com/teratron/gonn/pkg/neuron"
	"github.com/teratron/gonn/pkg/utils"
)

// Network
type Network[T utils.Float] struct {
	// All working neurons.
	cells []neuron.Neuron[T]

	// Input neurons.
	input *Bundle[T, *cell.Input[T]]

	// Output neurons.
	output *Bundle[T, *cell.Output[T]]

	// Hidden neurons.
	hidden *Bundle[T, *cell.Hidden[T]]
}

// NewNetwork
func NewNetwork[T utils.Float]() *Network[T] {
	return &Network[T]{
		cells:  make([]neuron.Neuron[T], 0),
		input:  NewBundle[T, *cell.Input[T]](),
		output: NewBundle[T, *cell.Output[T]](),
		hidden: NewBundle[T, *cell.Hidden[T]](),
	}
}
