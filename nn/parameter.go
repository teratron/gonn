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

//+-------------------------------------------------------------+
//| Neural network                                              |
//+-------------------------------------------------------------+
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

func (n *nn) Get(args ...Getter) Getter {
	if len(args) == 0 {
		//log.Printf("Return %T %v\n", args, args)
		Log("Return NN struct", true)
		return n
	} else {
		//fmt.Printf("--- %T %v\n", args[0], args[0])
		/*for _, v := range args {
			if s, ok := v.(Setter); ok {
				//fmt.Printf("--- %T %v\n", s, s)
				//s.Set(n)
			}
		}*/
	}
	return nil
}

//+-------------------------------------------------------------+
//| Neuron bias                                                 |
//+-------------------------------------------------------------+
// Initializing bias
func (b Bias) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		n.architecture.Set(b)
	}
}

func (n *nn) SetBias(bias Bias) {
	bias.Set(n)
}

// Getting bias
func (n *nn) Bias() (bias Bias) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		bias = v.Bias()
	}
	return
}

func (n *nn) GetBias() Bias {
	return n.Bias()
}

func (b Bias) Get(args ...Getter) Getter {
	//fmt.Printf("--- %T %v\n", args, args)
	return args[0]
}

// Checking bias
/*func (b Bias) Check() Checker {
	switch {
	case b < 0:
		return Bias(0)
	case b > 1:
		return Bias(1)
	default:
		return b
	}
}*/

//+-------------------------------------------------------------+
//| Learning rate                                               |
//+-------------------------------------------------------------+
// Initializing learning rate
func (r Rate) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		if rate, ok := r.Check().(Rate); ok {
			n.architecture.Set(rate)
		}
	}
}

func (n *nn) SetRate(rate Rate) {
	rate.Set(n)
}

// Getting learning rate
func (n *nn) Rate() (rate Rate) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		rate = v.Rate()
	}
	return
}

func (n *nn) GetRate() Rate {
	return n.Rate()
}

func (r Rate) Get(args ...Getter) Getter {
	return args[0]
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
// Initializing level loss

// Getting level loss

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
func (h Hidden) Set(args ...Setter) {
	if n, ok := args[0].(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			v.Set(h)
		}
	}
}

func (n *nn) SetHiddenLayer(args ...hidden) {
	if v, ok := n.architecture.(NeuralNetwork); ok {
		v.SetHiddenLayer(args...)
	}
}

func (h Hidden) Get(args ...Getter) Getter {
	panic("implement me")
}

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

func HiddenLayer(args ...hidden) Hidden {
	return args
}

