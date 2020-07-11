package nn

// Setter
/*func (n *neuron) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(n)
	}
}*/

// Initer
func (n *neuron) init(args ...Setter) bool {
	/*if v, ok := getArchitecture(set[0]); ok {
		v.Set(n)
	}*/
	return true
}
