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
func New(reader ...io.Reader) NeuralNetwork {
	if len(reader) > 0 {
		switch reader[0].(type) {
		case jsonType:

		//case xml:
		//case db:
		default:
		}
	} else {
	}
	n := &nn{
		Architecture: &perceptron{},
		IsInit:       false,
		IsTrain:      false,
	}
	n.Perceptron()
	return n
}

func (n *nn) init(input []float64, target ...[]float64) bool {
	if a, ok := n.Get().(NeuralNetwork); ok {
		n.IsInit = a.init(input, target...)
	}
	return n.IsInit
}

func (f floatType) Set(...Setter) {}

func (f floatType) Get(...Getter) GetterSetter {
	return nil
}

/*func (f floatArrayType) Set(...Setter) {}

func (f floatArrayType) Get(...Getter) GetterSetter {
	return nil
}*/