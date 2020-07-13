// Hopfield Neural Network - under construction
package nn

import "log"

/*type Hopfield interface {
	Hopfield() NeuralNetwork
}*/

type hopfield struct {
	Architecture
	Processor

	neuron			[]*neuron
	axon			[][]*axon
}

type hopfieldNeuron struct {
	energy floatType
}

// Returns a new Hopfield neural network instance with the default parameters
func (n *nn) Hopfield() NeuralNetwork {
	n.architecture = &hopfield{
		Architecture: n,
	}
	return n
}

// Setter
func (h *hopfield) Set(set ...Setter) {
	switch v := set[0].(type) {
	default:
		Log("This type of variable is missing for Hopfield Neural Network", false)
		log.Printf("\tset: %T %v\n", v, v) // !!!
	}
}

// Getter
func (h *hopfield) Get(set ...Setter) Getter {
	switch set[0].(type) {
	default:
		Log("This type of variable is missing for Hopfield Neural Network", false)
		log.Printf("\tget: %T %v\n", set[0], set[0]) // !!!
		return nil
	}
}

// Initialization
func (r *hopfield) init(args ...GetterSetter) bool {
	return true
}

// Calculating
func (r *hopfield) calc(args ...Initer) {
}

//
/*func (h *hopfieldNeuron) calc(args ...Initer) {
}*/

func (h *hopfieldNeuron) Set(args ...Setter) {
}

func (h *hopfieldNeuron) Get(args ...Setter) Getter {
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