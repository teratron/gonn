package nn

import "fmt"

type (
	LossModeUint   uint8   // Average error mode
	LossLimitFloat float64 // Level loss
)

const (
	// ModeMSE - Mean Squared Error
	ModeMSE uint8 = iota

	// ModeRMSE - Root Mean Squared Error
	ModeRMSE

	// ModeARCTAN - Arctan
	ModeARCTAN
)

// LossMode
func LossMode(mode ...uint8) LossModeUint {
	if len(mode) > 0 {
		return LossModeUint(mode[0])
	}
	return 0
}

// LossLimit
func LossLimit(level ...float64) LossLimitFloat {
	if len(level) > 0 {
		return LossLimitFloat(level[0])
	}
	return 0
}

// Set
func (l LossModeUint) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Set(l.check())
		}
	} else {
		LogError(fmt.Errorf("%w set for loss mode", ErrEmpty))
	}
}

// Set
func (l LossLimitFloat) Set(args ...Setter) {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			n.Set(l)
		}
	} else {
		LogError(fmt.Errorf("%w set for loss limit", ErrEmpty))
	}
}

// Get
func (l LossModeUint) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(l)
		}
	} else {
		return l
	}
	return nil
}

// Get
func (l LossLimitFloat) Get(args ...Getter) GetSetter {
	if len(args) > 0 {
		if n, ok := args[0].(NeuralNetwork); ok {
			return n.Get(l)
		}
	} else {
		return l
	}
	return nil
}

// check
func (l LossModeUint) check() LossModeUint {
	switch {
	case l < 0 || l > LossModeUint(ModeARCTAN):
		return LossModeUint(ModeMSE)
	default:
		return l
	}
}
