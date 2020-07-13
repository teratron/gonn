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
func (r rateType) Set(args ...Setter) {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			if c, ok := r.check().(rateType); ok {
				v.Set(c)
			}
		}
	}
}

// Getter
func (r rateType) Get(args ...Setter) Getter {
	if a, ok := args[0].(Architecture); ok {
		if v, ok := getArchitecture(a); ok {
			return v.Get(r)
		}
	}
	return nil
}

// Checker
func (r rateType) check() Getter {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}