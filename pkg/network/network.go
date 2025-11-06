package network

import (
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Network
type Network[T utils.Float] struct {
	// All working neurons.
	//Cells []neuron.Neuron[T]

	// Input neurons.
	Input Bundle[T, *cell.Input[T]] `json:"input" yaml:"input" toml:"input" xml:"input"`

	// Output neurons.
	Output Bundle[T, *cell.Output[T]] `json:"output" yaml:"output" toml:"output" xml:"output"`

	// Hidden neurons.
	Hidden Bundle[T, *cell.Hidden[T]] `json:"hidden" yaml:"hidden" toml:"hidden" xml:"hidden"`
}

// NewNetwork
func New[T utils.Float]() Network[T] {
	return Network[T]{
		//Cells:  make([]neuron.Neuron[T], 0),
		Input:  NewBundle[T, *cell.Input[T]](),
		Output: NewBundle[T, *cell.Output[T]](),
		Hidden: NewBundle[T, *cell.Hidden[T]](),
	}
}
