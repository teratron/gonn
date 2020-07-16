//
package nn

type (
	lowerRangeType floatType
	upperRangeType floatType
)

func LowerRange(lower ...lowerRangeType) GetterSetter {
	if len(lower) == 0 {
		return lowerRangeType(0)
	} else {
		return lower[0]
	}
}

func UpperRange(upper ...upperRangeType) GetterSetter {
	if len(upper) == 0 {
		return upperRangeType(0)
	} else {
		return upper[0]
	}
}

// Setter
func (l lowerRangeType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			if c, ok := l.check().(lowerRangeType); ok {
				a.Get().Set(c)
			}
		}
	}
}

func (u upperRangeType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			if c, ok := u.check().(upperRangeType); ok {
				a.Get().Set(c)
			}
		}
	}
}

// Getter
func (l lowerRangeType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return floatType(l)
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	}
	return nil
}

func (u upperRangeType) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		return floatType(u)
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(u)
		}
	}
	return nil
}

// Checker
func (l lowerRangeType) check() GetterSetter {
	switch {
	case l < 0 || l > 1:
		return DefaultRate
	default:
		return l
	}
}

func (u upperRangeType) check() GetterSetter {
	switch {
	case u < 0 || u > 1:
		return DefaultRate
	default:
		return u
	}
}