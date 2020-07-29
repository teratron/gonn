// Radial Basis Neural Network - under construction
package nn

import (
	"log"
)

type radialBasis struct {
	Architecture
	Constructor
}

// Returns a new Radial Basis neural network instance with the default parameters
func (n *NN) RadialBasis() NeuralNetwork {
	n.Architecture = &radialBasis{
		Architecture: n,
	}
	return n
}

// Setter
func (r *radialBasis) Set(args ...Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		default:
			Log("This type of variable is missing for Radial Basis Neural Network", true)
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (r *radialBasis) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		default:
			Log("This type of variable is missing for Radial Basis Neural Network", true)
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return r
	}
}

// Initialization
func (r *radialBasis) init(args ...Setter) bool {
	return true
}

// Train
/*func (r *radialBasis) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (r *radialBasis) Query(input []float64) []float64 {
	panic("implement me")
}*/