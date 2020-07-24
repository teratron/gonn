// Neuron bias
package nn

type biasType bool

func Bias(bias ...bool) GetterSetter {
	if len(bias) > 0 {
		return biasType(bias[0])
	} else {
		return biasType(false)
	}
}

// Setter
func (b biasType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(b)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (b biasType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(b)
		}
	} else {
		return b
	}
	return nil
}