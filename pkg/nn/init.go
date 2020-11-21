package nn

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/zigenzoog/gonn/pkg"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance
func New(reader ...pkg.Reader) NeuralNetwork {
	/*n := &nn{
		//Architecture: &architecture{},
		//isInit:  false,
		//isTrain: false,
		//json:    "",
	}*/
	n := new(nn)
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case Architecture:
			//n.Architecture = r
			//r.setArchitecture(n)
			n.setArchitecture(r)
		case Filer:
			r.Read(n)
		default:
			errNN(fmt.Errorf("%T %w for neural network", r, ErrMissingType))
		}
	} else {
		n.setArchitecture(Perceptron())
		//n.Architecture = Perceptron()
		//n.Architecture.setArchitecture(n)
	}
	return n
}

// init
func (n *nn) init(lenInput int, lenTarget ...interface{}) bool {
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		n.isInit = a.init(lenInput, lenTarget...)
	}
	return n.isInit
}

// getRand return random number from -0.5 to 0.5
func getRand() (r floatType) {
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
