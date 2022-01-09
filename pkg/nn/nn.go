package nn

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/arch"
)

// NeuralNetwork interface.
type NeuralNetwork interface {
	pkg.NeuralNetwork
}

// Floater interface.
type Floater interface {
	pkg.Floater
}

// New returns a new neural network instance.
func New(reader ...string) NeuralNetwork {
	if len(reader) > 0 {
		return arch.Get(reader[0])
	}
	return arch.Get(arch.Perceptron)
}

// ReadWeight reads and return array of weights.
func ReadWeight(name string) (weights pkg.Floater, err error) {
	return
}
