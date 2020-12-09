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
			var n NeuralNetwork
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
			}
			//fmt.Printf("Filer %T %s\n", r, r)

			var data interface{}
			if err = json.Unmarshal(b, &data); err != nil {
				LogError(fmt.Errorf("read unmarshal %w", err))
			}
			if value, ok := data.(map[string]interface{})["name"]; ok {
				n = getArchitecture(value.(string))
				if err = json.Unmarshal(b, &n); err != nil {
					LogError(fmt.Errorf("read unmarshal %w", err))
				}
				fmt.Println(n.(*perceptron).Weights)
				fmt.Printf("%T %v\n", n.(*perceptron).Weights, n.(*perceptron).Weights)
				//n.setName("asd")
				//n.Set(NeuronBias(false))

				n.setStateInit(false)
				n.setNameJSON(filename)
				if weight, ok := data.(map[string]interface{})["weights"]; ok && weight != nil /*&& len(w) > 0*/ {
					if w, ok := weight.(Float3Type); ok {
						fmt.Println(w)
					}
					//fmt.Println(weight)
					//fmt.Printf("%T %v\n", weight, weight)
				}
			}
			return n
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
		return Perceptron() //&perceptron{}
	case hopfieldName:
		return Hopfield() //&hopfield{}
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

// getLengthData returns the length of the slices
/*func getLengthData(data ...[]float64) []interface{} {
	var tmp []interface{}
	defer func() {
		tmp = nil
	}()
	if len(data) > 0 {
		for _, v := range data {
			tmp = append(tmp, len(v))
		}
	}
	return tmp
}*/
