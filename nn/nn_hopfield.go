// under construction
package nn

type hopfield struct {
	Architecture
	Parameter
}

// Initializing Hopfield Neural Network
func (n *nn) Hopfield() NeuralNetwork {
	n.architecture = &hopfield{}
	return n
}

// Setter
func (h *hopfield) Set(args ...Setter) {
}

// Getter
func (h *hopfield) Get(args ...Getter) Getter {
	return args[0]
}