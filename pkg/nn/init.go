// Initialization
package nn

import (
	"io"
	"math/rand"
	"time"
)

type floatType float32

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	Log("Start", false) // !!!
}

// New returns a new neural network instance with the default parameters
func New(init ...io.Reader) NeuralNetwork {
	if len(init) > 0 {

	} else {

	}
	n := &nn{
		Architecture: &perceptron{},
		isInit:       false,
		isTrain:      false,
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