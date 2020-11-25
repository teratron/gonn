package nn

import (
	"fmt"
	"math"

	"github.com/teratron/gonn/pkg"
)

type activationModeUint uint8

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
func ActivationMode(mode ...uint8) pkg.GetSetter {
	if len(mode) > 0 {
		return activationModeUint(mode[0])
	}
	return activationModeUint(0)
}

// Set
func (m activationModeUint) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(m.check())
		}
	} else {
		pkg.LogError(fmt.Errorf("%w set for activation", pkg.ErrEmpty))
	}
}

// Get
func (m activationModeUint) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(m)
		}
	} else {
		return m
	}
	return nil
}

// check
func (m activationModeUint) check() activationModeUint {
	switch {
	case m < 0 || m > activationModeUint(ModeTANH):
		return activationModeUint(ModeSIGMOID)
	default:
		return m
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
