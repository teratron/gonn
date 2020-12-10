package nn

import (
	"fmt"
	"math"
)

type ActivationModeUint uint8

const (
	// ModeLINEAR - Linear/identity
	ModeLINEAR uint8 = iota

	// ModeRELU - ReLu (rectified linear unit)
	ModeRELU

	// ModeLEAKYRELU - Leaky ReLu (leaky rectified linear unit)
	ModeLEAKYRELU

	// ModeSIGMOID - Logistic, a.k.a. sigmoid or soft step
	ModeSIGMOID

	// ModeTANH - TanH (hyperbolic tangent)
	ModeTANH
)

// ActivationMode
func ActivationMode(mode ...uint8) ActivationModeUint {
	if len(mode) > 0 {
		return ActivationModeUint(mode[0])
	}
	return 0
}

// Set
func (a ActivationModeUint) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Set(a.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for activation", ErrEmpty))
	}
}

// Get
func (a ActivationModeUint) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(a)
		}
	} else {
		return a
	}
	return nil
}

// check
func (a ActivationModeUint) check() ActivationModeUint {
	switch {
	case a < 0 || a > ActivationModeUint(ModeTANH):
		return ActivationModeUint(ModeSIGMOID)
	default:
		return a
	}
}

// calcActivation activation function
func calcActivation(value float64, mode uint8) float64 {
	switch mode {
	default:
		fallthrough
	case ModeLINEAR:
		return value
	case ModeRELU:
		switch {
		case value < 0:
			return 0
		default:
			return value
		}
	case ModeLEAKYRELU:
		switch {
		case value < 0:
			return .01 * value
		default:
			return value
		}
	case ModeSIGMOID:
		return 1 / (1 + math.Exp(-value))
	case ModeTANH:
		value = math.Exp(2 * value)
		if math.IsInf(value, 1) {
			return 1
		}
		return (value - 1) / (value + 1)
	}
}

// calcDerivative derivative activation function
func calcDerivative(value float64, mode uint8) float64 {
	switch mode {
	default:
		fallthrough
	case ModeLINEAR:
		return 1
	case ModeRELU:
		switch {
		case value < 0:
			return 0
		default:
			return 1
		}
	case ModeLEAKYRELU:
		switch {
		case value < 0:
			return .01
		default:
			return 1
		}
	case ModeSIGMOID:
		return value * (1 - value)
	case ModeTANH:
		return 1 - math.Pow(value, 2)
	}
}
