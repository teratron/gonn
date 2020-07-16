// Initialization
package nn

import (
	"math/rand"
	"time"
)

type Initer interface {
	init(...Setter) bool
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
	if len(args) > 0 {
		if a, ok := n.Get().(NeuralNetwork); ok {
			n.isInit = a.init(args...)
		}
	} else {
		Log("Empty init()", true) // !!!
	}
	return n.isInit
}

/*func (f floatType) init(...Setter) bool {
	return true
}
 */
func (f floatArrayType) init(...Setter) bool {
	return true
}