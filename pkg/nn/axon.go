package nn

type axon struct {
	weight  floatType			//
	synapse map[string]Getter	//
}

/*func Axon() GetterSetter {
	return &axon{}
}

// Setter
func (a *axon) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Get().Set(a)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (a *axon) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(a)
		}
	} else {
		return a
	}
	return nil
}*/