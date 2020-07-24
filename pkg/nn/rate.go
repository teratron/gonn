// Learning rate
package nn

type rateType floatType

// Default learning rate
const DefaultRate float32 = .3

func Rate(rate ...float32) GetterSetter {
	if len(rate) > 0 {
		return rateType(rate[0])
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
		return r
	}
	return nil
}

// Checking
func (r rateType) check() rateType {
	switch {
	case r < 0 || r > 1:
		return rateType(DefaultRate)
	default:
		return r
	}
}