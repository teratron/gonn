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

func (h *hopfield) Set(args ...Setter) {
}

func (h *hopfield) Get(args ...Getter) Getter {
	panic("implement me")
}