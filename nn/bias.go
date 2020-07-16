// Neuron bias
package nn

type biasType bool

func Bias(bias ...biasType) GetterSetter {
	if len(bias) == 0 {
		return biasType(false)
	} else {
		return bias[0]
	}
}

// Setter
func (b biasType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(b)
		}
	}
}

// Getter
func (b biasType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return b
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(b)
		}
	}
	return nil
}