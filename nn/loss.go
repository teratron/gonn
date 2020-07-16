// Loss
package nn

type (
	lossType      floatArrayType
	modeLossType  uint8     // Average error mode
	levelLossType floatType // Level loss
)

const (
	ModeMSE      modeLossType  = 0      // Mean Squared Error
	ModeRMSE     modeLossType  = 1      // Root Mean Squared Error
	ModeARCTAN   modeLossType  = 2      // Arctan
	MinLevelLoss levelLossType = 10e-33 // The minimum value of the error limit at which training is forcibly terminated
)

func (n *nn) Loss(target []float64) (loss float64) {
	if n.isInit && n.isQuery {
		//if a, ok := getArchitecture(n); ok {
			//loss = n.Get().Loss(target)
			//fmt.Println(n.Get())
		//}
	} else {
		Log("An uninitialized neural network", true)
	}
	return
}

func Loss(target []float64) Initer {
	return lossType(target)
}

func ModeLoss(mode ...modeLossType) GetterSetter {
	if len(mode) == 0 {
		return modeLossType(0)
	} else {
		return mode[0]
	}
}

func LevelLoss(level ...levelLossType) GetterSetter {
	if len(level) == 0 {
		return levelLossType(0)
	} else {
		return level[0]
	}
}

// Setter
func (m modeLossType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			if c, ok := m.check().(modeLossType); ok {
				a.Get().Set(c)
			}
		}
	}
}

func (l levelLossType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			if c, ok := l.check().(levelLossType); ok {
				a.Get().Set(c)
			}
		}
	}
}

// Getter
func (m modeLossType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return m
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(m)
		}
	}
	return nil
}

func (l levelLossType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return floatType(l)
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	}
	return nil
}

// Checker
func (m modeLossType) check() GetterSetter {
	switch {
	case m < 0 || m > ModeARCTAN:
		return ModeMSE
	default:
		return m
	}
}

func (l levelLossType) check() GetterSetter {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}

// Initer
func (l lossType) init(...Setter) bool {
	return true
}