package architecture

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
	"github.com/teratron/gonn/nn/hopfield"
	"github.com/teratron/gonn/nn/perceptron"
)

// Architecture
/*type Architecture interface {
	Get() Architecture
	Set(Architecture)
}*/

/*type architecture struct {
	Architecture
}*/

/*func (n *nn) architecture() Architecture {
	return n.Architecture
}*/

/*func (n *nn) setArchitecture(network Architecture) {
	n.Architecture = network
	n.Architecture.setArchitecture(n)
}*/

type NeuralNetwork struct {
	//gonn.NN
	//perceptron.Parameter
	//perceptron.NeuralNetwork
}

/*
type NeuralNetworkH struct {
	//gonn.NN
	//hopfield.NeuralNetwork
}*/

const (
	Perceptron = perceptron.Name
	Hopfield   = hopfield.Name
)

// Architecture
func Get(name ...string) gonn.Architecture {
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

/*func (nn *NeuralNetwork) Get() Architecture {
	return nil
}*/
