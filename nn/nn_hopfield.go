//
package nn

type hopfield struct {
	Architecture
	Parameter
}

func (n *NN) Hopfield() NeuralNetwork {
	n.architecture = &hopfield{}
	return n
}

func (h *hopfield) Set(args ...Setter) {
}

func (h *hopfield) Get(args ...Getter) Getter {
	panic("implement me")
}