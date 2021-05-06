package nn

import (
	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/zoo"
)

// NeuralNetwork.
type NeuralNetwork interface {
	pkg.NeuralNetwork
}

// Floater.
type Floater interface {
	pkg.Floater
}

// New returns a new neural network instance.
func New(reader ...string) NeuralNetwork {
	if len(reader) > 0 {
		return zoo.Get(reader[0])
	}
	return zoo.Get(zoo.Perceptron)
}
