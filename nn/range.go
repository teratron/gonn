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
			a.Get().Set(l)
		}
	}
}

func (u upperRangeType) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", true) // !!!
	} else {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(u)
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