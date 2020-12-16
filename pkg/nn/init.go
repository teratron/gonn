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
					n.(*perceptron).initHiddenFromWeight()
					n.(*perceptron).lenInputFromWeight()
					n.(*perceptron).lenOutputFromWeight()
					n.(*perceptron).initNeuronFromWeight()

					//fmt.Printf("%T %v\n", n.(*perceptron), n.(*perceptron))
					//fmt.Printf("%T %v\n", n.(*perceptron).Weights, n.(*perceptron).Weights)
					//fmt.Println(len(n.(*perceptron).Weights),cap(n.(*perceptron).Weights))
					//fmt.Println(len(n.(*perceptron).Weights[0]),cap(n.(*perceptron).Weights[0]))
					//fmt.Println(len(n.(*perceptron).Weights[1]),cap(n.(*perceptron).Weights[1]))
					//fmt.Println(len(n.(*perceptron).Weights[1][0]),cap(n.(*perceptron).Weights[1][0]))

					//fmt.Printf("%T %v\n", n.(*perceptron).Hidden, n.(*perceptron).Hidden)
					//fmt.Println(n.Weight().Length())
					/*bias := 0
					if n.NeuronBias() {
						bias = 1
					}*/
					//length := n.Weight().Length() - 1
					//fmt.Println(length, len(n.(*perceptron).Hidden))
					/*n.SetHiddenLayer() = make(HiddenArrUint, length)
					for i := range p.Hidden {
						p.Hidden[i] = uint(len(p.Weights[i]))
					}*/

					n.setNameJSON(filename)
					n.setStateInit(true)
					/*if weight, ok := data.(map[string]interface{})["weights"]; ok && weight != nil {
						//fmt.Printf("%T %v\n", weight, weight)
						if w, ok := weight.(Float3Type); ok {
							fmt.Printf("%T %v\n", w, w)
						}
						//fmt.Println(weight)
						//fmt.Printf("%T %v\n", weight, weight)
					}*/
				}
				return n
			}
			//fmt.Printf("Filer %T %s\n", r, r)
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
