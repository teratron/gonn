package network

import (
	"github.com/teratron/gonn/pkg/neuron/axon"
	"github.com/teratron/gonn/pkg/neuron/cell"
	"github.com/teratron/gonn/pkg/utils"
)

// Network
type Network[T utils.Float] struct {
	Rate T

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

func (n *Network[T]) Build() error {
	// Создаем связи между Input -> первый Hidden
	for _, i := range n.Input.cells {
		for _, h := range n.Hidden.cells {
			h.IncomingAxons = append(h.IncomingAxons, axon.New(i, h))
		}
	}

	// Создаем связи между Hidden слоями и Output
	// ... аналогично

	return nil
}
