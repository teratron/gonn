//
package nn

type neuron struct {
	value    floatType
	axon     []*axon
	specific GetterSetter	// feature
}

func Neuron() Initer {
	return &neuron{}
}

// Setter
func (n *neuron) Set(args ...Setter) {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			v.Set(n)
		}
	}
}

// Getter
func (n *neuron) Get(args ...Setter) Getter {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			return v.Get(n)
		}
	}
	return nil
}

// Initialization
func (n *neuron) init(args ...Setter) bool {
	return true
}

// Calculating
func (n *neuron) calc(args ...Initer) Getter {
	return nil
}