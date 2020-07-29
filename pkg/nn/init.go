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
		//case csv:
		//case db:
		default:
		}
	} else {
	}
	n := &NN{
		Architecture: &perceptron{},
		IsInit:       false,
		IsTrain:      false,
	}
	n.Perceptron() //???
	return n
}

func (n *NN) init(input []float64, target ...[]float64) bool {
	if a, ok := n.Get().(NeuralNetwork); ok {
		n.IsInit = a.init(input, target...)
	}
	return n.IsInit
}

func (f floatType) Set(...Setter) {}
func (f floatType) Get(...Getter) GetterSetter { return nil }

// Return random number from -0.5 to 0.5
func getRand() (r floatType) {
	r = 0
	for r == 0 {
		r = floatType(rand.Float64() - .5)
	}
	return
}