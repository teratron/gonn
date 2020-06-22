// Learning rate
package nn

type rateType float32

// Default learning rate
const DefaultRate rateType = .3

func Rate(rate ...rateType) Setter {
	if len(rate) == 0 {
		return rateType(0)
	} else {
		return rate[0]
	}
}

// Setter
func (r rateType) Set(set ...Setter) {
	if v, ok := getArchitecture(set[0]); ok {
		if c, ok := r.Check().(rateType); ok {
			v.Set(c)
		}
	}
}

// Getter
func (r rateType) Get(set ...Setter) Getter {
	if v, ok := getArchitecture(set[0]); ok {
		return v.Get(r)
	}
	return nil
}

// Checker
func (r rateType) Check() Getter {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}