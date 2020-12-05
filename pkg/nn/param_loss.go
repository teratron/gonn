package nn

import "fmt"

type (
	lossModeUint   uint8   // Average error mode
	lossLimitFloat float64 // Level loss
)

const (
	// ModeMSE - Mean Squared Error
	ModeMSE uint8 = iota

	// ModeRMSE - Root Mean Squared Error
	ModeRMSE

	// ModeARCTAN - Arctan
	ModeARCTAN

	// MinLossLimit the minimum value of the error limit at which training is forcibly terminated
	MinLossLimit float64 = 10e-33
)

// LossMode
func LossMode(mode ...uint8) GetSetter {
	if len(mode) > 0 {
		return lossModeUint(mode[0])
	}
	return lossModeUint(0)
}

// LossLimit
func LossLimit(level ...float64) GetSetter {
	if len(level) > 0 {
		return lossLimitFloat(level[0])
	}
	return lossLimitFloat(0)
}

// Set
func (l lossModeUint) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for loss mode", ErrEmpty))
	}
}

// Set
func (l lossLimitFloat) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for loss limit", ErrEmpty))
	}
}

// Get
func (l lossModeUint) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return l
	}
	return nil
}

// Get
func (l lossLimitFloat) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return l
	}
	return nil
}

// check
func (l lossModeUint) check() lossModeUint {
	switch {
	case l < 0 || l > lossModeUint(ModeARCTAN):
		return lossModeUint(ModeMSE)
	default:
		return l
	}
}

// check
func (l lossLimitFloat) check() lossLimitFloat {
	switch {
	case l < 0 || l < lossLimitFloat(MinLossLimit):
		return lossLimitFloat(MinLossLimit)
	default:
		return l
	}
}
