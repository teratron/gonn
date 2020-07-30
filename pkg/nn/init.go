// Initialization
package nn

import (
	"io"
	"math/rand"
	"time"

	"github.com/zigenzoog/gonn/pkg"
)

type floatType float32

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance with the default parameters
func New(reader ...io.Reader) NeuralNetwork {
	if len(reader) > 0 {
		switch reader[0].(type) {
		case jsonType:
		//case xmlType:
		//case csvType:
		//case dbType:
		default:
		}
	} else {
	}
	n := &NN{
		Architecture: &perceptron{},
		isInit:       false,
		isTrain:      false,
	}
	n.Perceptron() //???
	return n
}

func (n *NN) init(input []float64, target ...[]float64) bool {
	if a, ok := n.Get().(NeuralNetwork); ok {
		n.isInit = a.init(input, target...)
	}
	return n.isInit
}

func (f floatType) Set(...pkg.Setter) {}
func (f floatType) Get(...pkg.Getter) pkg.GetterSetter { return nil }

// Return random number from -0.5 to 0.5
func getRand() (r floatType) {
	r = 0
	for r == 0 {
		r = floatType(rand.Float64() - .5)
	}
	return
}