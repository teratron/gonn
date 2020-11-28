package nn

import "github.com/teratron/gonn/pkg"

type biasBool bool

// NeuronBias
func NeuronBias(bias ...bool) pkg.GetSetter {
	if len(bias) > 0 {
		return biasBool(bias[0])
	}
	return biasBool(false)
}

// Set
func (b biasBool) Set(args ...pkg.Setter) {
	/*if len(args) > 0 {
		if n, ok := args[0].(*nn); ok && !n.isInit {
			n.Get().Set(b)
		}
	} else {
		pkg.LogError(fmt.Errorf("%w set for bias", pkg.ErrEmpty))
	}*/
}

// Get
func (b biasBool) Get(args ...pkg.Getter) pkg.GetSetter {
	/*if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get().Get(b)
		}
	} else {
		return b
	}*/
	return nil
}
