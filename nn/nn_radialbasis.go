//
package nn

type radialBasis struct {
	Architecture
	Parameter
}

// Initializing Radial Basis Neural Network
func (n *NN) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{}
	return n
}

func (r *radialBasis) Set(args ...Setter) {
}

func (r *radialBasis) Get(args ...Getter) Getter {
	panic("implement me")
}