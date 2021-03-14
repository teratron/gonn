package architecture

import (
	"github.com/teratron/gonn/hopfield"
)

func Architecture(name string) gonn.NeuralNetwork {
	switch name {
	//case perceptronName:
	//return Perceptron()
	case hopfield.HopfieldName:
		return hopfield.Hopfield()
	default:
		//log.Println(fmt.Errorf("get architecture: neural network is %w", ErrNotRecognized))
		return nil
	}
}
