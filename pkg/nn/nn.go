package nn

import (
	"github.com/teratron/gonn/pkg"
	arch "github.com/teratron/gonn/pkg/architecture"
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
