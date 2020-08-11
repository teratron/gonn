// Initialization
package nn

import (
	"math/rand"
	"time"

	"github.com/zigenzoog/gonn/pkg"
)

type floatType float32

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance with the default parameters
func New(reader ...pkg.Reader) NeuralNetwork {
	//n := new(nn)
	n := &nn{
		//Parameter: make(map[string]Architecture),
		isInit:    false,
		IsTrain:   false,
		json:      "",
		xml:       "",
		csv:       "",
		db:        "",
	}
	//n.Parameter = make(map[string]interface{})
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case *perceptron:
			n.Architecture = r
			//n.Parameter["Architecture"] = r
			//n.Parameters = r
			n.perceptron()
		/*case jsonType:
			if len(r) > 0 {
				//n =
			} else {
				log.Fatal("Отсутствует название файла нейросети")
			}
		case xmlType:
		case csvType:
		case dbType:*/
		default:
		}
	} else {
		n.Architecture = &perceptron{}
		//n.Parameter["perceptron"] = &perceptron{}
		n.perceptron()
	}
	return n
}

func reset() *nn {
	n := &nn{
		Architecture: nil,
		isInit:    false,
		IsTrain:   false,
		json:      "",
		xml:       "",
		csv:       "",
		db:        "",
	}
	return n
}

func (n *nn) init(lenInput int, lenTarget ...interface{}) bool {
	if a, ok := n.Get().(NeuralNetwork); ok {
		n.isInit = a.init(lenInput, lenTarget...)
	}
	return n.isInit
}

func (f floatType) Set(...pkg.Setter) {}
func (f floatType) Get(...pkg.Getter) pkg.GetterSetter {
	return nil
}

// getRand return random number from -0.5 to 0.5
func getRand() (r floatType) {
	r = 0
	for r == 0 {
		r = floatType(rand.Float64() - .5)
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