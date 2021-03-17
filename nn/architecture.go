package nn

import (
	"github.com/teratron/gonn"
	zoo "github.com/teratron/gonn/architecture"
)

func architecture(name ...string) gonn.Architecture {
	/*return &NeuralNetwork{
		NeuralNetwork: zoo.Get(name[0]),
	}*/
	return zoo.Get(name[0])
}

// Perceptron
func Perceptron() gonn.Architecture {
	/*return &NeuralNetwork{
		NeuralNetwork: zoo.Get(zoo.Perceptron),
	}*/
	return zoo.Get(zoo.Perceptron)
}

// Hopfield
func Hopfield() gonn.Architecture {
	/*return &NeuralNetwork{
		NeuralNetwork: zoo.Get(zoo.Hopfield),
	}*/
	return zoo.Get(zoo.Hopfield)
}
