package nn

import "fmt"

type RateFloat FloatType

// Default learning rate
const DefaultRate float32 = .3

// Rate
func LearningRate(rate ...float32) RateFloat {
	if len(rate) > 0 {
		return RateFloat(rate[0])
	}
	return 0
}

// Set
func (r RateFloat) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Set(r.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for rate", ErrEmpty))
	}
}

// Get
func (r RateFloat) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(r)
		}
	} else {
		return r
	}
	return nil
}

// check
func (r RateFloat) check() RateFloat {
	switch {
	case r < 0 || r > 1:
		return RateFloat(DefaultRate)
	default:
		return r
	}
}
