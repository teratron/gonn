// Initialization
package nn

import (
	"math/rand"
	"time"
)

type (
	floatType      float32
	//floatArrayType []float64
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	Log("Start", false) // !!!
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	n := &nn{
		Architecture:	&perceptron{},
		isInit:			false,
		isTrain:		false,
	}
	n.Perceptron()
	return n
}

func (n *nn) init(input []float64, target ...[]float64) bool {
	if a, ok := n.Get().(NeuralNetwork); ok {
		n.isInit = a.init(input, target...)
	}
	return n.isInit
}

func (f floatType) Set(...Setter) {}

func (f floatType) Get(...Getter) GetterSetter {
	return nil
}

/*func (f floatArrayType) Set(...Setter) {}

func (f floatArrayType) Get(...Getter) GetterSetter {
	return nil
}*/