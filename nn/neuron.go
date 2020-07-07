package nn

// Setter
func (n *neuron) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(n)
	}
}
