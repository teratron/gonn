//
package nn

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Getter) GetterSetter
}

type Setter interface {
	Set(...Setter)
}

// Setter
func (n *nn) Set(args ...Setter) {
	if len(args) > 0 {
		for _, v := range args {
			if s, ok := v.(Setter); ok {
				s.Set(n)
			}
		}
	} else {
		Log("Empty Set()", true) // !!!
	}
}

func (a *app) Set(args ...Setter) {
	if len(args) > 0 {
		/*for _, v := range args {
			if s, ok := v.(Setter); ok {
				s.Set(n)
			}
		}*/
	} else {
		Log("Empty Set()", true) // !!!
	}
}

// Getter
func (n *nn) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		for _, v := range args {
			if g, ok := v.(Getter); ok {
				return g.Get(n)
			}
		}
	} else {
		if a, ok := n.Architecture.(NeuralNetwork); ok {
			return a
		}
	}
	return nil
}

func (a *app) Get(args ...Getter) GetterSetter {
	if len(args) > 0 {
		/*for _, v := range args {
			if g, ok := v.(Getter); ok {
				return g.Get(n)
			}
		}*/
	} else {
		/*if a, ok := a; ok {
			return a
		}*/
	}
	return nil
}