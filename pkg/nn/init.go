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
			if value, ok := r.getValue("name").(string); ok {
				n := getArchitecture(value)
				n.Read(r)
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
		LogError(fmt.Errorf("neural network is %w", ErrNotRecognized))
		return nil
	}
}

// getRand return random number from -0.5 to 0.5
func getRandFloat() (r float64) {
	for r == 0 || r > .5 || r < -.5 {
		r = rand.Float64() - .5
	}
	return
}

func Debug() {
	//fmt.Println(randFloat())
	/*randFloat = func() floatType {
		return 42
	}*/
	//fmt.Println(randFloat())
}
