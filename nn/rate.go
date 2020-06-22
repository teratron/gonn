// Learning rate
package nn

type RateType float32

// Default learning rate
const DefaultRate RateType = .3

func Rate(rate ...RateType) RateType {
	return rate[0]
}

// Setter
func (r RateType) Set(rate ...Setter) {
	if n, ok := rate[0].(*nn); ok {
		if rate, ok := r.Check().(RateType); ok {
			n.architecture.Set(rate)
		}
	}
}

// Getter
func (r RateType) Get(rate ...Getter) Getter {
	return rate[0]
}

// Initializing
func (n *nn) SetRate(rate RateType) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		v.SetRate(rate)
	}
}

// Return
func (n *nn) GetRate() (rate RateType) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		rate = v.GetRate()
	}
	return
}

// Checking
func (r RateType) Check() Checker {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}