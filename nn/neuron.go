//
package nn

type neuron struct {
	value    floatType
	axon     []*axon
	specific Getter
}

func Neuron() Initer {
	return &neuron{}
}

// Setter
func (n *neuron) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(n)
		}
	}
}

// Getter
func (n *neuron) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return n
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(n)
		}
	}
	return nil
}

// Initialization
func (n *neuron) init(...Setter) bool {
	return true
}

// Calculating
func (n *neuron) calc(...Initer) Getter {
	return nil
}