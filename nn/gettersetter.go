package nn

import (
	"fmt"
	"math/rand"
)

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

// Setter
func (n *nn) Set(args ...Setter) {
	if len(args) == 0 {
		Log("Empty Set()", false)
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
		fmt.Printf("%T %v\n", n.Architecture.(NeuralNetwork), n.Architecture.(NeuralNetwork))
		return n.Architecture.(NeuralNetwork)
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

func (f floatType) Get(...Setter) Getter {
	return nil
}

func (f floatArrayType) Set(...Setter) {}

func (f floatArrayType) Get(...Setter) Getter {
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