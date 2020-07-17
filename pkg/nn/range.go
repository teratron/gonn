//
package nn

type (
	lowerRangeType float64
	upperRangeType float64
)

func LowerRange(lower ...lowerRangeType) GetterSetter {
	if len(lower) > 0 {
		return lower[0]
	} else {
		return lowerRangeType(0)
	}
}

func UpperRange(upper ...upperRangeType) GetterSetter {
	if len(upper) > 0 {
		return upper[0]
	} else {
		return upperRangeType(0)
	}
}

// Setter
func (l lowerRangeType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(l)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

func (u upperRangeType) Set(args ...Setter) {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			a.Get().Set(u)
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (l lowerRangeType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(l)
		}
	} else {
		return floatType(l)
	}
	return nil
}

func (u upperRangeType) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		if a, ok := args[0].(NeuralNetwork); ok {
			return a.Get().Get(u)
		}
	} else {
		return floatType(u)
	}
	return nil
}