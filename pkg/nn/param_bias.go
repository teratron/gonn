package nn

type biasBool bool

// NeuronBias
func NeuronBias(bias ...bool) GetSetter {
	if len(bias) > 0 {
		return biasBool(bias[0])
	}
	return biasBool(false)
}

// Set
func (b biasBool) Set(args ...Setter) {
	/*if len(args) > 0 {
		if n, ok := args[0].(*nn); ok && !n.isInit {
			n.Get().Set(b)
		}
	} else {
		LogError(fmt.Errorf("%w set for bias", ErrEmpty))
	}*/
}

// Get
func (b biasBool) Get(args ...Getter) GetSetter {
	/*if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(b)
		}
	} else {
		return b
	}*/
	return nil
}
