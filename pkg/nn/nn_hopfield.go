// Hopfield Neural Network - under construction
package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

type hopfield struct {
	Architecture
	Constructor

	neuron []*Neuron
	axon   [][]*Axon
}

/*func Hopfield() io.ReadWriter {
	return nil
}*/

// Returns a new Hopfield neural network instance with the default parameters
func (n *nn) Hopfield() NeuralNetwork {
	n.Architecture = &hopfield{
		Architecture: n,
	}
	return n
}

// Setter
func (h *hopfield) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (h *hopfield) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Hopfield Neural Network", true)
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return h
	}
}

// Initialization
func (h *hopfield) init(args ...pkg.Setter) bool {
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