// Initialization
package nn

import (
	"math/rand"
	"time"
)

type Initer interface {
	init(...Setter) bool
	GetterSetter
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	Log("Start", false) // !!!
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	n := &nn{
		Architecture:	&perceptron{},
		isInit:			false,
		isQuery:		false,
		isTrain:		false,
		language:		"en",
		logging:		1,
	}
	n.Perceptron()
	return n
}

func (n *nn) init(args ...Setter) bool {
	if v, ok := getArchitecture(n); ok {
		n.isInit = v.init(args...)
	}
	return n.isInit
}

/*func (f floatType) init(args ...Setter) bool {
	return true
}*/