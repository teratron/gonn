package network

import (
	"github.com/teratron/gonn/pkg/neuron"
	cell2 "github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Network
type Network[T utils.Float] struct {
	// All working neurons.
	Cells []neuron.Neuron[T]

	// Input neurons.
	Input Bundle[T, *cell2.Input[T]]

	// Output neurons.
	Output Bundle[T, *cell2.Output[T]]

	// Hidden neurons.
	Hidden Bundle[T, *cell2.Hidden[T]]
}

// NewNetwork
func New[T utils.Float]() Network[T] {
	return Network[T]{
		Cells:  make([]neuron.Neuron[T], 0),
		Input:  NewBundle[T, *cell2.Input[T]](),
		Output: NewBundle[T, *cell2.Output[T]](),
		Hidden: NewBundle[T, *cell2.Hidden[T]](),
	}
}
