package nn

import (
	"math/rand"
)

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

// Getter
func (n *nn) Get(args ...Getter) GetterSetter {
	if len(args) == 0 {
		if a, ok := n.Architecture.(GetterSetter); ok {
			return a
		}
	} else {
		for _, v := range args {
			if g, ok := v.(Getter); ok {
				return g.Get(n)
			}
		}
	}
	return nil
}

func (f floatType) Set(...Setter) {}

func (f floatType) Get(...Getter) GetterSetter {
	return nil
}

func (f floatArrayType) Set(...Setter) {}

func (f floatArrayType) Get(...Getter) GetterSetter {
	return nil
}

// Return random number from -0.5 to 0.5
func getRand() (r floatType) {
	r = 0
	for r == 0 {
		r = floatType(rand.Float64() - .5)
	}
	return
}