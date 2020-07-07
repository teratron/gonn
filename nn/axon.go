package nn

// Setter
func (a *axon) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(a)
	}
}
