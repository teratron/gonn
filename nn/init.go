//
package nn

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	Log("Start", false)
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	n := &nn{
		architecture:	&perceptron{},
		isInit:			false,
		isTrain:		false,
		lastNeuron:		0,
		lastAxon:		0,
		language:		"en",
		logging: 		true,
	}
	n.Perceptron()
	return n
}

// Init
// data[0] - input data
// data[1] - target data
// ... - any data
func (n *nn) init(data ...[]float64) bool {
//func (n *nn) Init(input, target []float64) bool {
	if v, ok := getArchitecture(n); ok {
		n.isInit = v.init(data...)
	}
	return true
}