// Neuron bias
package nn

import "fmt"

type Bias bool

type BiasInt interface {
	//Perceptron
	//
	Bias() Bias
	GetBias() Bias
	SetBias(Bias)
}

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
	fmt.Printf("%T %v\n", n.architecture.(BiasInt), n.architecture.(BiasInt))
	if v, ok := n.architecture.(BiasInt); ok {
		fmt.Printf("ttttt %T %v\n", v, v)

		return v.Bias()

		//return v.Get(v.Bias())

	}
	return false
}

func (n *nn) GetBias() Bias {
	return n.Bias()
}