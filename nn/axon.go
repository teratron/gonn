package nn

func Axon() Initer {
	return &axon{}
}

// Setter
func (a *axon) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(a)
	}
}

/*func (s *synapse) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(s)
	}
}*/

// Getter
func (a *axon) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(a)
	}
	return nil
}

// Initialization
func (a *axon) init(args ...GetterSetter) bool {
	return true
}