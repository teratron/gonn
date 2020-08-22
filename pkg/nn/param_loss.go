// Loss
package nn

import "github.com/zigenzoog/gonn/pkg"

type (
	lossModeType  uint8   // Average error mode
	lossLevelType float64 // Level loss
)

const (
	ModeMSE      uint8   = iota   // Mean Squared Error
	ModeRMSE                      // Root Mean Squared Error
	ModeARCTAN                    // Arctan
	MinLossLevel float64 = 10e-33 // The minimum value of the error limit at which training is forcibly terminated
)

func LossMode(mode ...uint8) pkg.GetterSetter {
	if len(mode) > 0 {
		return lossModeType(mode[0])
	} else {
		return lossModeType(0)
	}
}

func LossLevel(level ...float64) pkg.GetterSetter {
	if len(level) > 0 {
		return lossLevelType(level[0])
	} else {
		return lossLevelType(0)
	}
}

func (n *NN) LossMode() uint8 {
	return n.Architecture.(Parameter).LossMode()
}

func (n *NN) LossLevel() float64 {
	return n.Architecture.(Parameter).LossLevel()
}

// Set
func (m lossModeType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(m.check())
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

func (l lossLevelType) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Get
func (m lossModeType) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(m)
		}
	} else {
		return m
	}
	return nil
}

func (l lossLevelType) Get(args ...pkg.Getter) pkg.GetterSetter {
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
func (m lossModeType) check() lossModeType {
	switch {
	case m < 0 || m > lossModeType(ModeARCTAN):
		return lossModeType(ModeMSE)
	default:
		return m
	}
}

func (l lossLevelType) check() lossLevelType {
	switch {
	case l < 0 || l < lossLevelType(MinLossLevel):
		return lossLevelType(MinLossLevel)
	default:
		return l
	}
}