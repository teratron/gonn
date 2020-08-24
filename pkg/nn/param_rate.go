// Learning rate
package nn

import "github.com/zigenzoog/gonn/pkg"

type rateType floatType

// Default learning rate
const DefaultRate float32 = .3

func Rate(rate ...float32) pkg.GetSetter {
	if len(rate) > 0 {
		return rateType(rate[0])
	} else {
		return rateType(0)
	}
}

func (n *NN) Rate() float32 {
	return n.Architecture.(Parameter).Rate()
}

// Setter
func (r rateType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(r.check())
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (r rateType) Get(args ...pkg.Getter) pkg.GetSetter {
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