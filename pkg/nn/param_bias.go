// Neuron bias
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

type biasType bool

func Bias(bias ...bool) pkg.GetSetter {
	if len(bias) > 0 {
		return biasType(bias[0])
	} else {
		return biasType(false)
	}
}

func (n *NN) Bias() bool {
	return n.Architecture.(Parameter).Bias()
}

// Set
func (b biasType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*NN); ok && !n.IsInit {
			n.Get().Set(b)
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Get
func (b biasType) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(b)
		}
	} else {
		return b
	}
	return nil
}