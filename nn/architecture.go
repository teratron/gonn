package nn

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/hopfield"
)

const (
	HopfieldName   = "hopfield"
	PerceptronName = "perceptron"
)

func architecture(name string) gonn.NeuralNetwork {
	switch name {
	case PerceptronName:
		return Perceptron()
	case HopfieldName:
		return Hopfield()
	default:
		log.Println(fmt.Errorf("get architecture: neural network is %w", ErrNotRecognized))
		return nil
	}
}

// Perceptron return perceptron neural network
func Perceptron() *perceptron {
	return &perceptron{
		Name:       perceptronName,
		Activation: ModeSIGMOID,
		Loss:       ModeMSE,
		Limit:      .1,
		Rate:       DefaultRate,
	}
}

// Hopfield return
func Hopfield() *hopfield.Hopfield {
	return &hopfield.Hopfield{
		Name: HopfieldName,
	}
}
