package nn

import (
	"fmt"
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
			//fmt.Println("Filer")
			var n NeuralNetwork
			//r.Read(n)
			fmt.Println(n)
			return n
		default:
			pkg.LogError(fmt.Errorf("%T %w for neural network", r, pkg.ErrMissingType))
			return nil
		}
	}
	return Perceptron()
}

// getRand return random number from -0.5 to 0.5
func getRand() (r pkg.FloatType) {
	for r == 0 {
		r = pkg.FloatType(rand.Float64() - .5)
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
