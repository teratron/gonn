package nn

import "fmt"

type rateFloat FloatType

// Default learning rate
const DefaultRate float32 = .3

// Rate
func LearningRate(rate ...float32) GetSetter {
	if len(rate) > 0 {
		return rateFloat(rate[0])
	}
	return rateFloat(0)
}

// Set
func (r rateFloat) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(r.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for rate", ErrEmpty))
	}
}

// Get
func (r rateFloat) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(r)
		}
	} else {
		return r
	}
	return nil
}

// check
func (r rateFloat) check() rateFloat {
	switch {
	case r < 0 || r > 1:
		return rateFloat(DefaultRate)
	default:
		return r
	}
}
