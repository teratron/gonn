package nn

import (
	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/zoo"
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
func New(reader ...string) pkg.NeuralNetwork {
	if len(reader) > 0 {
		return zoo.Get(reader[0])
	}
	return zoo.Get(zoo.Perceptron)
}
