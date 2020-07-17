// Learning rate
package nn

type rateType float32

// Default learning rate
const DefaultRate rateType = .3

func Rate(rate ...rateType) GetterSetter {
	if len(rate) > 0 {
		return rate[0]
	} else {
		return rateType(0)
	}
}

// Setter
func (r rateType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(r.check())
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (r rateType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(r)
		}
	} else {
		return floatType(r)
	}
	return nil
}

// Checking
func (r rateType) check() rateType {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}