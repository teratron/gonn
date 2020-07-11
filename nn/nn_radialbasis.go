// Radial Basis Neural Network - under construction
package nn

import (
	"log"
)

/*type RadialBasis interface {
	RadialBasis() NeuralNetwork
}*/

type radialBasis struct {
	Architecture
	Processor
}

type radialBasisNeuron struct {
	error			floatType
}

// Initializing Radial Basis Neural Network
func (n *nn) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{
		Architecture:	n,
	}
	return n
}

// Setter
func (r *radialBasis) Set(set ...Setter) {
	switch v := set[0].(type) {
	default:
		Log("This type of variable is missing for Radial Basis Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v) // !!!
	}
}

// Getter
func (r *radialBasis) Get(set ...Setter) Getter {
	switch set[0].(type) {
	default:
		Log("This type of variable is missing for Radial Basis Neural Network", false)
		log.Printf("\tget: %T %v\n", set[0], set[0]) // !!!
		return nil
	}
}

// Initer

//
func (r *radialBasisNeuron) calc(args ...Initer) {
}

// Train
/*func (r *radialBasis) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (r *radialBasis) Query(input []float64) []float64 {
	panic("implement me")
}*/