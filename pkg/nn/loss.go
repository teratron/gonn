// Loss
package nn

type (
	lossType      []float64
	modeLossType  uint8					// Average error mode
	levelLossType floatType				// Level loss
)

const (
	ModeMSE      modeLossType  = 0      // Mean Squared Error
	ModeRMSE     modeLossType  = 1      // Root Mean Squared Error
	ModeARCTAN   modeLossType  = 2      // Arctan
	MinLevelLoss levelLossType = 10e-33 // The minimum value of the error limit at which training is forcibly terminated
)

func (n *nn) Loss(target []float64) (loss float64) {
	if n.isInit && n.isQuery {
	} else {
		Log("An uninitialized neural network", true)
	}
	return
}

func Loss(target []float64) GetterSetter {
	return lossType(target)
}

func ModeLoss(mode ...modeLossType) GetterSetter {
	if len(mode) > 0 {
		return mode[0]
	} else {
		return modeLossType(0)
	}
}

func LevelLoss(level ...levelLossType) GetterSetter {
	if len(level) > 0 {
		return level[0]
	} else {
		return levelLossType(0)
	}
}

// Setter
func (l lossType) Set(args ...Setter) {}

func (m modeLossType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(m.check())
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

func (l levelLossType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l.check())
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (l lossType) Get(args ...Getter) GetterSetter {
	return nil
}

func (m modeLossType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(m)
		}
	} else {
		return m
	}
	return nil
}

func (l levelLossType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return floatType(l)
	}
	return nil
}

// Checking
func (m modeLossType) check() modeLossType {
	switch {
	case m < 0 || m > ModeARCTAN:
		return ModeMSE
	default:
		return m
	}
}

func (l levelLossType) check() levelLossType {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}