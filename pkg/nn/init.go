package nn

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/teratron/gonn/pkg"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated
const MaxIteration int = 10e+05

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance
func New(reader ...pkg.Reader) NeuralNetwork {
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case NeuralNetwork:
			return r
		case Filer:
			var n NeuralNetwork
			r.Read(n)
			return n
		default:
			errNN(fmt.Errorf("%T %w for neural network", r, ErrMissingType))
			return nil
		}
	}
	return Perceptron()
}

// init
/*func (n *nn) init(lenInput int, lenTarget ...interface{}) bool {
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		n.isInit = a.init(lenInput, lenTarget...)
	}
	return n.isInit
}*/

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
