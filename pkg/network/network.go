package network

import (
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Network
type Network[T utils.Float] struct {
	// Input neurons.
	Input bundle[T, *cell.Input[T]] `json:"input" toml:"input" xml:"input"`

	// Output neurons.
	Output bundle[T, *cell.Output[T]] `json:"output" toml:"output" xml:"output"`

	// Hidden neurons.
	Hidden bundle[T, *cell.Hidden[T]] `json:"hidden" toml:"hidden" xml:"hidden"`
}

// New
func New[T utils.Float]() Network[T] {
	return Network[T]{
		Input:  newBundle[T, *cell.Input[T]](),
		Output: newBundle[T, *cell.Output[T]](),
		Hidden: newBundle[T, *cell.Hidden[T]](),
	}
}
