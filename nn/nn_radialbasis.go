// Radial Basis Neural Network - under construction
package nn

type radialBasis struct {
	Architecture
	Parameter
}

// Initializing Radial Basis Neural Network
func (n *nn) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{}
	return n
}

// Setter
func (r *radialBasis) Set(args ...Setter) {
}

// Getter
func (r *radialBasis) Get(args ...Getter) Getter {
	return args[0]
}