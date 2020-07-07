package nn

import (
	"math/rand"
	"time"
)

// Setter
func (n *axon) Set(set ...Setter) {
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
	for _, a := range n.axon {
		//if b, ok := a.synapse["bias"]; !ok || (ok && b.(biasType) == true) {
		a.weight = randWeight()
		//}
	}
}
