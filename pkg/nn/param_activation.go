package nn

import (
	"fmt"
	"math"

	"github.com/zigenzoog/gonn/pkg"
)

type activationModeUint uint8

const (
	ModeLINEAR    uint8 = iota // Linear/identity
	ModeRELU                   // ReLu - rectified linear unit
	ModeLEAKYRELU              // Leaky ReLu - leaky rectified linear unit
	ModeSIGMOID                // Logistic, a.k.a. sigmoid or soft step
	ModeTANH                   // TanH - hyperbolic tangent
)

// ActivationMode
func ActivationMode(mode ...uint8) pkg.GetSetter {
	if len(mode) > 0 {
		return activationModeUint(mode[0])
	} else {
		return activationModeUint(0)
	}
}

// ActivationMode
func (n *nn) ActivationMode() uint8 {
	return n.Architecture.(Parameter).ActivationMode()
}

// Set
func (m activationModeUint) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
			a.Get().Set(m.check())
		}
	} else {
		errNN(fmt.Errorf("%w set for activation", ErrEmpty))
	}
}

// Get
func (m activationModeUint) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(Architecture); ok {
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