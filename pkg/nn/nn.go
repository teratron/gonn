package nn

import (
	arch "github.com/teratron/gonn/pkg/architecture"
	"github.com/teratron/gonn/pkg/perceptron"
)

// NeuralNetwork interface.
/*type NeuralNetwork interface {
	pkg.NeuralNetwork
}*/

// Floater interface.
/*type Floater interface {
	Floater
}*/

// New returns a new neural network instance.
func New[T float32 | float64](reader ...string) perceptron.NN[T] {
	if len(reader) > 0 {
		return arch.Get(reader[0])
	}

	return arch.Get(arch.Perceptron)
}
