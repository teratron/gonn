//
package nn


func Neuron() Initer {
	return &neuron{}
}

// Setter
func (n *neuron) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(n)
	}
}

// Getter
func (n *neuron) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(n)
	}
	return nil
}

// Initialization
func (n *neuron) init(args ...GetterSetter) bool {
	return true
}

// Calculating
func (n *neuron) calc(args ...Initer) {
	//network.calc
}