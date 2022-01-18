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

		_ = copy(nn.input, input)

		if nn.Weights[0][0][0] != 0 {
			_ = copy(nn.weight, nn.Weights)
		} else if nn.weight[0][0][0] != 0 {
			_ = copy(nn.Weights, nn.weight)
		}

		nn.calcNeuron()
		for i, n := range nn.neuron[nn.lastLayerIndex] {
			nn.output[i] = float64(n.value)
		}
		return nn.output
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("query: %v\n", err)
	return nil
}
