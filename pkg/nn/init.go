package nn

import (
	"fmt"
	"log"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance.
func New(reader ...Reader) NeuralNetwork {
	if len(reader) > 0 {
		var err error
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case Filer:
			switch v := r.getValue("name").(type) {
			case string:
				n := getArchitecture(v)
				if err = n.Read(r); err == nil {
					return n
				}
			case error:
				err = v
			}
		default:
			err = fmt.Errorf("%T %w", r, ErrMissingType)
		}
		if err != nil {
			err = fmt.Errorf("new: %w", err)
			log.Println(err)
		}
		return nil
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
		log.Println(fmt.Errorf("get architecture: neural network is %w", ErrNotRecognized))
		return nil
	}
}

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration int = 10e+05

var maxIteration = getMaxIteration

// getMaxIteration
func getMaxIteration() int {
	return MaxIteration
}

var randFloat = getRandFloat

// getRand return random number from -0.5 to 0.5.
func getRandFloat() (r float64) {
	for r == 0 || r > .5 || r < -.5 {
		r = rand.Float64() - .5
	}
	return
}
