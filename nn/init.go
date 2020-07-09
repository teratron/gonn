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
		language:		"en",
		logging: 		1,
	}
	n.Perceptron()
	return n
}

// Init
func (n *nn) init(args ...Setter) bool {
	if v, ok := getArchitecture(n); ok {
		n.isInit = v.init(args...)
	}
	return true
}