package nn

type axon struct {
	weight  floatType				//
	synapse map[string]GetterSetter	//
}

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
func (a *axon) init(...Setter) bool {
	return true
}

// Calculating
func (a *axon) calc(...Initer) Getter {
	return nil
}