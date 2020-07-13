// Neuron bias
package nn

type biasType bool

func Bias(bias ...biasType) Setter {
	if len(bias) == 0 {
		return biasType(false)
	} else {
		return bias[0]
	}
}

// Setter
func (b biasType) Set(args ...Setter) {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			v.Set(b)
		}
	}

}

// Getter
func (b biasType) Get(args ...Setter) Getter {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			return v.Get(b)
		}
	}
	return nil
}