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
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Get().Set(a)
		}
	}
}

// Getter
func (a *axon) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return a
	} else {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(a)
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