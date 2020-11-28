package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance
func New(reader ...pkg.Reader) NeuralNetwork {
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case pkg.Filer:
			//fmt.Printf("Filer %T %s\n", r, r)
			var n NeuralNetwork
			filename := r.ToString()
			if len(filename) == 0 {
				pkg.LogError(fmt.Errorf("json: file json is missing"))
			}
			b, err := ioutil.ReadFile(filename)
			if err != nil {
				pkg.LogError(err)
			}
			var data interface{}
			if err = json.Unmarshal(b, &data); err != nil {
				pkg.LogError(fmt.Errorf("read unmarshal %w", err))
			}
			if value, ok := data.(map[string]interface{})["name"]; ok {
				fmt.Println(data.(map[string]interface{})["hidden"])
				fmt.Println(data.(map[string]interface{})["weights"])
				//fmt.Println(getArchitecture(value.(string)))
				n = getArchitecture(value.(string))
				//fmt.Println(n)
				if err = json.Unmarshal(b, &n); err != nil {
					pkg.LogError(fmt.Errorf("read unmarshal %w", err))
				}
				n.setStateInit(false)
				n.setNameJSON(filename)
			}
			return n
		default:
			pkg.LogError(fmt.Errorf("%T %w for neural network", r, pkg.ErrMissingType))
			return nil
		}
	}
	return Perceptron()
}

// getArchitecture
func getArchitecture(name string) NeuralNetwork {
	switch name {
	case perceptronName:
		return &perceptron{}
	case hopfieldName:
		return &hopfield{}
	default:
		pkg.LogError(fmt.Errorf("neural network %w", pkg.ErrNotFound))
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
func getLengthData(data ...[]float64) []interface{} {
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
}
