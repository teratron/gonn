//
package nn

import (
	"math/rand"
	"time"
)

const (
	MaxIteration uint32	= 10e+05 // The maximum number of iterations after which training is forcibly terminated

	ModeMSE      uint8	= 0      // Mean Squared Error
	ModeRMSE     uint8	= 1      // Root Mean Squared Error
	ModeARCTAN   uint8	= 2      // Arctan
)

func init() {
	Log("Start", false)
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	return &nn{
		architecture:	&perceptron{},
		language:		"en",
		logging: 		true,
	}
}

// The function fills all weights with random numbers from -0.5 to 0.5
func (n *nn) setRandWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	randWeight := func() (r floatType) {
		r = 0.
		for r == 0 {
			r = floatType(rand.Float64() - .5)
		}
		return
	}
	for _, a := range n.axon {
		if b, ok := a.synapse["bias"]; !ok || (ok && b.(biasType) == true) {
			a.weight = randWeight()
		}
	}
}

//
func (n *nn) Init(input, target []floatType) bool {
	return true
}