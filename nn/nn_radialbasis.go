//
package nn

type radialBasis struct {
	Architecture
	Parameter
}

func (n *NN) RadialBasis() NeuralNetwork {
	n.architecture = &radialBasis{}
	return n
}

func (r *radialBasis) Set(args ...Setter) {
}

func (r *radialBasis) Get(args ...Getter) Getter {
	panic("implement me")
}