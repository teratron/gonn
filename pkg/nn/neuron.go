package nn

import "github.com/zigenzoog/gonn/pkg"

type neuron struct {
	value    floatType	// Neuron value
	axon     []*axon	// All incoming axons
	specific pkg.Getter
}

/*func Neuron() GetterSetter {
	return &neuron{}
}*/

// Set
func (n *neuron) Set(args ...pkg.Setter) {
	/*if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(n)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}*/
}

// Get
func (n *neuron) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(n)
		}
	} else {
		return n
	}
	return nil
}