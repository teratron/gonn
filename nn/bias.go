// Neuron bias
package nn

type Bias bool

// Setter
func (b Bias) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		n.architecture.Set(b)
	}
}

// Getter
func (b Bias) Get(args ...Getter) Getter {
	return args[0]
}

// Initializing
func (n *nn) SetBias(bias Bias) {
	bias.Set(n)
}

// Return
func (n *nn) Bias() (bias Bias) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		bias = v.Bias()
	}
	return
}

func (n *nn) GetBias() Bias {
	return n.Bias()
}