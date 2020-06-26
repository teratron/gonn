// Loss
package nn

type (
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
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		Log("An uninitialized neural network", true)
	}
	return
}

func Loss(mode ...modeLossType) Setter {
	if len(mode) == 0 {
		return modeLossType(0)
	} else {
		return mode[0]
	}
}

func LevelLoss(level ...levelLossType) Setter {
	if len(level) == 0 {
		return levelLossType(0)
	} else {
		return level[0]
	}
}

// Setter
func (m modeLossType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		if c, ok := m.Check().(modeLossType); ok {
			v.Set(c)
		}
	}
}

func (l levelLossType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		if c, ok := l.Check().(levelLossType); ok {
			v.Set(c)
		}
	}
}

// Getter
func (m modeLossType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(m)
	}
	return nil
}

func (l levelLossType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(l)
	}
	return nil
}

// Checker
func (m modeLossType) Check() Getter {
	switch {
	case m < 0 || m > ModeARCTAN:
		return ModeMSE
	default:
		return m
	}
}

func (l levelLossType) Check() Getter {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}