// Learning rate
package nn

type rateType float32

// Default learning rate
const DefaultRate rateType = .3

func Rate(rate ...rateType) GetterSetter {
	if len(rate) == 0 {
		return rateType(0)
	} else {
		return rate[0]
	}
}

// Setter
func (r rateType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			if c, ok := r.check().(rateType); ok {
				a.Get().Set(c)
			}
		}
	}
}

// Getter
func (r rateType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return floatType(r)
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(r)
		}
	}
	return nil
}

// Checker
func (r rateType) check() GetterSetter {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}