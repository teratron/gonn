// Radial Basis Neural Network - under construction
package nn

type RadialBasis interface {
	RadialBasis() NeuralNetwork
}

type radialBasis struct {
	cup int
	Architecture
	//Parameter
}

// Initializing Radial Basis Neural Network
func (n *nn) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{
		cup: 1,
	}
	return n
}

// Setter
func (r *radialBasis) Set(args ...Setter) {
}

// Getter
func (r *radialBasis) Get(args ...Getter) Getter {
	switch args[0].(type) {
	/*case Bias:
		return r.bias
	case Rate:
		return r.rate
	case Hidden:
		return r.hiddenLayer*/
	default:
		return nil
	}
}