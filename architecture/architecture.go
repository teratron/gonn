package architecture

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/nn/hopfield"
	"github.com/teratron/gonn/nn/perceptron"
)

const (
	Perceptron = perceptron.Title
	Hopfield   = hopfield.Title
)

func Architecture(name ...string) gonn.NeuralNetwork {
	if len(name) > 0 {
		switch name[0] {
		case Perceptron:
			return perceptron.Perceptron()
		case Hopfield:
			return hopfield.Hopfield()
		default:
			log.Println(fmt.Errorf("get architecture: neural network is %w", gonn.ErrNotRecognized))
			return nil
		}
	}
	return perceptron.Perceptron()
}
