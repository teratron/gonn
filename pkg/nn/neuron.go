//
package nn

//type neuronType [][]*neuron

type neuron struct {
	value    floatType
	axon     []*axon
	specific Getter
}

/*func Neuron() GetterSetter {
	return &neuron{}
}*/

// Setter
func (n *neuron) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(n)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (n *neuron) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(n)
		}
	} else {
		return n
	}
	return nil
}

/*func (n *neuronType) Set(args ...Setter) {
}

func (n *neuronType) Get(args ...Getter) GetterSetter {
	return nil
}*/