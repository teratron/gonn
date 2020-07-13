package nn

func Axon() Initer {
	return &axon{}
}

// Setter
func (a *axon) Set(args ...Setter) {
	if n, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(n); ok {
			v.Set(a)
		}
	}
}

/*func (s *synapse) Set(args ...Setter) {
if a, ok := args[0].(Architecture); ok {
if v, ok := getArchitecture(args[0]); ok {
		v.Set(s)
	}
	}
}*/

// Getter
func (a *axon) Get(args ...Setter) Getter {
	if n, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(n); ok {
			return v.Get(a)
		}
	}
	return nil
}

// Initialization
func (a *axon) init(args ...GetterSetter) bool {
	return true
}

// Calculating
func (a *axon) calc(args ...Initer) GetterSetter {
	return nil
}