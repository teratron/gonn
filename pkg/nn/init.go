package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
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
			filename := r.ToString()
			if len(filename) == 0 {
				LogError(fmt.Errorf("init: file config is missing"))
			}

			b, err := ioutil.ReadFile(filename)
			if err != nil {
				LogError(err)
			}

			switch r.(type) {
			case jsonString:
				var data interface{}
				if err = json.Unmarshal(b, &data); err != nil {
					LogError(fmt.Errorf("read unmarshal %w", err))
				}
				var n NeuralNetwork
				if value, ok := data.(map[string]interface{})["name"]; ok {
					n = getArchitecture(value.(string))
					if err = json.Unmarshal(b, &n); err != nil {
						LogError(fmt.Errorf("read unmarshal %w", err))
					}
					//fmt.Println(len(n.(*perceptron).Weights),cap(n.(*perceptron).Weights))
					//fmt.Printf("%T %v\n", n.(*perceptron).Weights, n.(*perceptron).Weights)
					if n.(*perceptron).Weights != nil && len(n.(*perceptron).Weights) > 0 {
						n.(*perceptron).setLenInputFromWeight()
						n.(*perceptron).setLenOutputFromWeight()
						n.(*perceptron).initHiddenFromWeight()
						n.(*perceptron).initNeuronFromWeight()
						n.setStateInit(true)
					}
					n.setNameJSON(filename)
				}
				return n
			default:
				//LogError(fmt.Errorf("%T %w for neural network", r, ErrMissingType))
				//return nil
			}
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
func getRand() (r FloatType) {
	for r == 0 {
		r = FloatType(rand.Float64() - .5)
	}
	return
}
