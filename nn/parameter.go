package nn

type Parameter interface {
	Bias() Bias
	GetBias() Bias
	SetBias(Bias)

	Rate() Rate
	GetRate() Rate
	SetRate(Rate)

	GetNumHiddenLayer() hidden
	GetHiddenLayer() Hidden
	SetHiddenLayer(...hidden)
}

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Getter) Getter
}

type Setter interface {
	Set(...Setter)
}

type Checker interface {
	Check() Checker
}

type (
	Bias			bool
	Rate			float32
	Loss			Float
	hidden			uint16
	Hidden			[]hidden
)

//+-------------------------------------------------------------+
//| Neural network                                              |
//+-------------------------------------------------------------+
// Setter
func (n *nn) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty set", true)
	} else {
		for _, v := range args {
			if s, ok := v.(Setter); ok {
				s.Set(n)
			}
		}
	}
}

// Getter
func (n *nn) Get(args ...Getter) Getter {
	if len(args) == 0 {
		return n
	} else {
		for _, v := range args {
			if g, ok := v.(Getter); ok {
				return g.Get(n)
			}
		}
	}
	return nil
}

//+-------------------------------------------------------------+
//| Neuron bias                                                 |
//+-------------------------------------------------------------+
// Setter
func (b Bias) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		n.architecture.Set(b)
	}
}

// Getter
func (b Bias) Get(args ...Getter) Getter {
	return args[0]
}

// Initializing bias
func (n *nn) SetBias(bias Bias) {
	bias.Set(n)
}

// Return bias
func (n *nn) Bias() (bias Bias) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		bias = v.Bias()
	}
	return
}

func (n *nn) GetBias() Bias {
	return n.Bias()
}

//+-------------------------------------------------------------+
//| Learning rate                                               |
//+-------------------------------------------------------------+
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

// Initializing learning rate
func (n *nn) SetRate(rate Rate) {
	rate.Set(n)
}

// Return learning rate
func (n *nn) Rate() (rate Rate) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		rate = v.Rate()
	}
	return
}

func (n *nn) GetRate() Rate {
	return n.Rate()
}

// Checking learning rate
func (r Rate) Check() Checker {
	switch {
	case r < 0 || r > 1:
		return DefaultRate
	default:
		return r
	}
}

//+-------------------------------------------------------------+
//| Level loss                                        			|
//+-------------------------------------------------------------+
// Setter

// Getter

// Initializing level loss

// Return level loss

// Checking level loss
func (l Loss) Check() Checker {
	switch {
	case l < 0:
		return MinLevelLoss
	default:
		return l
	}
}

//+-------------------------------------------------------------+
//| Hidden layers							                    |
//+-------------------------------------------------------------+
func HiddenLayer(args ...hidden) Hidden {
	return args
}

func NumHiddenLayer(args ...hidden) hidden {
	return args[0]
}

// Setter
func (h Hidden) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			v.Set(h)
		}
	}
}

// Getter
func (h Hidden) Get(args ...Getter) Getter {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			return v.Get(h)
		}
	}
	return nil
}

func (h hidden) Get(args ...Getter) Getter {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			return v.Get(h)
		}
	}
	return nil
}

// Initializing hidden layers
func (n *nn) SetHiddenLayer(args ...hidden) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		v.SetHiddenLayer(args...)
	}
}

// Return hidden layers
func (n *nn) GetHiddenLayer() Hidden {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		return v.GetHiddenLayer()
	}
	return nil
}

func (n *nn) GetNumHiddenLayer() hidden {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		return v.GetNumHiddenLayer()
	}
	return 0
}
