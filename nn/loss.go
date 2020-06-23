// Level loss
package nn

type lossType floatType

// The minimum value of the error limit at which training is forcibly terminated
const MinLevelLoss lossType = 10e-33

func LevelLoss(loss ...lossType) Setter {
	if len(loss) == 0 {
		return lossType(0)
	} else {
		return loss[0]
	}
}

// Setter
func (l lossType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		if c, ok := l.Check().(lossType); ok {
			v.Set(c)
		}
	}
}

// Getter
func (l lossType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(l)
	}
	return nil
}

// Checker
func (l lossType) Check() Getter {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}