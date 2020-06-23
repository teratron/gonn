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
func (b biasType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		v.Set(b)
	}
}

// Getter
func (b biasType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(b)
	}
	return nil
}