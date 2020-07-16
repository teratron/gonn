// Hopfield Neural Network - under construction
package nn

import "log"

type Hopfield interface {
	Hopfield() NeuralNetwork
}

type hopfield struct {
	Architecture
	Processor

	neuron []*neuron
	axon   [][]*axon
}

type hopfieldNeuron struct {
	energy floatType
}

// Returns a new Hopfield neural network instance with the default parameters
func (n *nn) Hopfield() NeuralNetwork {
	n.Architecture = &hopfield{
		Architecture: n,
	}
	return n
}

// Setter
func (h *hopfield) Set(args ...Setter) {
	switch v := args[0].(type) {
	default:
		Log("This type of variable is missing for Hopfield Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v) // !!!
	}
}

// Getter
func (h *hopfield) Get(args ...Getter) GetterSetter {
	switch args[0].(type) {
	default:
		if len(args) == 0 { return h }
		Log("This type of variable is missing for Hopfield Neural Network", false)
		log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
		return nil
	}
}

// Initialization
func (h *hopfield) init(args ...Setter) bool {
	return true
}

// Calculating
func (h *hopfield) calc(args ...Initer) Getter {
	return nil
}

//
/*func (h *hopfieldNeuron) calc(args ...Initer) {
}*/

func (h *hopfieldNeuron) Set(args ...Setter) {
}

func (h *hopfieldNeuron) Get(args ...Getter) GetterSetter {
	return nil
}


// Train
/*func (h *hopfield) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (h *hopfield) Query(input []float64) []float64 {
	panic("implement me")
}*/