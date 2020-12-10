package nn

import "fmt"

type BiasBool bool

// NeuronBias
func NeuronBias(bias ...bool) BiasBool {
	if len(bias) > 0 {
		return BiasBool(bias[0])
	}
	return false
}

// Set
func (b BiasBool) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Set(b)
		}
	} else {
		LogError(fmt.Errorf("%w set for bias", ErrEmpty))
	}
}

// Get
func (b BiasBool) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(b)
		}
	} else {
		return b
	}
	return nil
}
