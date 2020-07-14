// Radial Basis Neural Network - under construction
package nn

import (
	"log"
)

type RadialBasis interface {
	RadialBasis() NeuralNetwork
}

type radialBasis struct {
	Architecture
	Processor
}

type radialBasisNeuron struct {
	error floatType
}

// Returns a new Radial Basis neural network instance with the default parameters
func (n *nn) RadialBasis() NeuralNetwork {
	n.Architecture = &radialBasis{
		Architecture: n,
	}
	return n
}

func (r *radialBasis) architecture() *radialBasis {
	return r
}

// Setter
func (r *radialBasis) Set(args ...Setter) {
	switch v := args[0].(type) {
	default:
		Log("This type of variable is missing for Radial Basis Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v) // !!!
	}
}

// Getter
func (r *radialBasis) Get(args ...Setter) Getter {
	switch args[0].(type) {
	default:
		Log("This type of variable is missing for Radial Basis Neural Network", false)
		log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
		return nil
	}
}

// Initialization
func (r *radialBasis) init(args ...Setter) bool {
	return true
}

// Calculating
func (r *radialBasis) calc(args ...Initer) Getter {
	return nil
}

//
/*func (r *radialBasisNeuron) calc(args ...Initer) {
}*/

func (r *radialBasisNeuron) Set(args ...Setter) {
}

func (r *radialBasisNeuron) Get(args ...Setter) Getter {
	return nil
}

// Train
/*func (r *radialBasis) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (r *radialBasis) Query(input []float64) []float64 {
	panic("implement me")
}*/