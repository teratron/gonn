package nn

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Setter) Getter
}

type Setter interface {
	Set(...Setter)
}

type Checker interface {
	Check() Getter
}

// Setter
func (n *nn) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty set", false)
	} else {
		for _, v := range args {
			if s, ok := v.(Setter); ok {
				s.Set(n)
			}
		}
	}
}

// Getter
func (n *nn) Get(args ...Setter) Getter {
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

func getArchitecture(set Setter) (NeuralNetwork, bool) {
	if n, ok := set.(*nn); ok {
		if v, ok := n.architecture.(NeuralNetwork); ok {
			return v, ok
		}
	}
	return nil, false
}