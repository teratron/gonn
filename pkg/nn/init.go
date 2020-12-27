package nn

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance
func New(reader ...Reader) NeuralNetwork {
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case Filer:
			//fmt.Printf("%T, %v", v, v)
			if value, ok := r.getValue("name").(string); ok {
				n := getArchitecture(value)
				r.Read(n)

				if n.(*perceptron).Weights != nil && len(n.(*perceptron).Weights) > 0 {
					n.(*perceptron).initFromWeight()
				}
				//n.setNameJSON(filename)
				return n
			}
			return nil
		default:
			LogError(fmt.Errorf("%T %w for neural network", r, ErrMissingType))
			return nil
		}
	}
	return Perceptron()
}

// getArchitecture
func getArchitecture(name string) NeuralNetwork {
	switch name {
	case perceptronName:
		return Perceptron()
	case hopfieldName:
		return Hopfield()
	default:
		LogError(fmt.Errorf("neural network %w", ErrNotFound))
		return nil
	}
}

// getRand return random number from -0.5 to 0.5
func getRand() (r floatType) {
	for r == 0 {
		r = floatType(rand.Float64() - .5)
	}
	return
}
