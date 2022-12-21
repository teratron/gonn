package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
)

// Query querying dataset.
func (nn *NN) Query(input []float64) []float64 {
	var err error
	if len(input) > 0 {
		nn.mutex.Lock()
		defer nn.mutex.Unlock()

		if !nn.isInit {
			err = pkg.ErrInit
			goto ERROR
		} else if nn.lenInput != len(input) {
			err = fmt.Errorf("invalid number of elements in the input data")
			goto ERROR
		}

		if nn.Weights[0][0][0] != 0 {
			nn.weights = nn.Weights
		}

		nn.input = pkg.ToFloat1Type(input)

		nn.calcNeurons()
		for i, n := range nn.neurons[nn.lastLayerIndex] {
			nn.output[i] = float64(n.value)
		}
		return nn.output
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("perceptron.NN.Query: %v\n", err)
	return nil
}
