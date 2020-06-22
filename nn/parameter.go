package nn

type Parameter interface {

/*	//
	Bias() Bias
	GetBias() Bias
	SetBias(Bias)

	//
	Rate() Rate
	GetRate() Rate
	SetRate(Rate)

	//
	GetHidden() Hidden
	SetHidden(...hidden)*/
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