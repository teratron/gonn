//
package nn

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
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

// The function fills all weights with random numbers from -0.5 to 0.5
func (n *nn) setRandWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	randWeight := func() (r floatType) {
		r = 0
		for r == 0 {
			r = floatType(rand.Float64() - .5)
		}
		return
	}
	for i, a := range n.axon {
		//if b, ok := a.synapse["bias"]; !ok || (ok && b.(biasType) == true) {
		//a.weight = randWeight()
		//}
		if i == 0 {
			fmt.Printf("%T %v\n", a, &a.weight)
			fmt.Printf("%T %v\n", n.axon, &n.axon[i].weight)
		}
	}
	for i := 0; i <= n.lastAxon; i++ {
		n.axon[i].weight = randWeight()
	}
	//n.axon[0].weight = randWeight()
	//fmt.Printf("%T %v\n", n.axon[0], n.axon[0])
	//fmt.Println(n.axon[0].weight)
}
