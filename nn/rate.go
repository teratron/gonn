// Learning rate
package nn

type Rate float32

// Default rate
const DefaultRate Rate = .3

// Setter
func (r Rate) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		if rate, ok := r.Check().(Rate); ok {
			n.architecture.Set(rate)
		}
	}
}

// Getter
func (r Rate) Get(args ...Getter) Getter {
	return args[0]
}

// Initializing
func (n *nn) SetRate(rate Rate) {
	rate.Set(n)
}

// Return
func (n *nn) Rate() (rate Rate) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		rate = v.Rate()
	}
	return
}

func (n *nn) GetRate() Rate {
	return n.Rate()
}

// Checking
func (r Rate) Check() Checker {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}