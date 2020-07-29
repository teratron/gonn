// Hopfield Neural Network - under construction
package nn

import (
	"log"
)

type Hopfield struct {
	Architecture
	Constructor

	neuron []*neuron
	axon   [][]*axon
}

// Returns a new Hopfield neural network instance with the default parameters
func (n *NN) Hopfield() NeuralNetwork {
	n.Architecture = &Hopfield{
		Architecture: n,
	}
	return n/*.Get().(*hopfield.Hopfield).Hopfield()*/
}

// Setter
func (h *Hopfield) Set(args ...Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		default:
			Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (h *Hopfield) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		default:
			Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return h
	}
}

// Initialization
func (h *Hopfield) init(args ...Setter) bool {
	return true
}

// Train
/*func (h *hopfield) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (h *hopfield) Query(input []float64) []float64 {
	panic("implement me")
}*/