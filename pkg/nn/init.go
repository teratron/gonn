package nn

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/zigenzoog/gonn/pkg"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

// New returns a new neural network instance with the default parameters
func New(reader ...pkg.Reader) NeuralNetwork {
	n := &nn{
		Architecture: &architecture{},
		IsInit:       false,
		IsTrain:      false,
		json:         "",
	}
	if len(reader) > 0 {
		switch r := reader[0].(type) {
		case Architecture:
			n.Architecture = r
			r.setArchitecture(n)
		case Filer:
			r.Read(n)
		default:
			errNN(fmt.Errorf("%T %w for neural network", r, ErrMissingType))
		}
	} else {
		n.Architecture = &perceptron{}
		n.Architecture.setArchitecture(n)
	}
	return n
}

// init
func (n *nn) init(lenInput int, lenTarget ...interface{}) bool {
	if a, ok := n.Architecture.(NeuralNetwork); ok {
		n.IsInit = a.init(lenInput, lenTarget...)
	}
	return n.IsInit
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

// Debug
func Debug(args ...interface{}) {
	log.Println(args[0])
}
