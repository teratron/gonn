// Neuron bias
package nn

import "github.com/zigenzoog/gonn/pkg"

type biasType bool

func Bias(bias ...bool) pkg.GetterSetter {
	if len(bias) > 0 {
		return biasType(bias[0])
	} else {
		return biasType(false)
	}
}

// Setter
func (b biasType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(*NN); ok && !n.isInit {
			n.Get().Set(b)
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (b biasType) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(b)
		}
	} else {
		return b
	}
	return nil
}