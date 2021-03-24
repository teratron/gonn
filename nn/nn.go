package nn

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/zoo"
)

// NeuralNetwork
type NeuralNetwork interface {
	gonn.NeuralNetwork
}

// Floater
type Floater interface {
	gonn.Floater
}

// New returns a new neural network instance.
func New(reader ...string) NeuralNetwork {
	if len(reader) > 0 {
		return zoo.Get(reader[0])
	}
	return zoo.Get(zoo.Perceptron)
}
