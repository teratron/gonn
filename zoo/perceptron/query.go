package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// Query querying dataset
func (nn *NN) Query(input []float64) (output []float64) {
	var err error
	if len(input) > 0 {
		if !nn.isInit {
			err = gonn.ErrInit
			goto ERROR
		} else if nn.lenInput != len(input) {
			err = fmt.Errorf("invalid number of elements in the input data")
			goto ERROR
		}

		nn.calcNeuron(input)
		output = make([]float64, nn.lenOutput)
		for i, n := range nn.neuron[nn.lastLayerIndex] {
			output[i] = n.value
		}
	} else {
		err = gonn.ErrNoInput
	}

ERROR:
	if err != nil {
		log.Println(fmt.Errorf("query: %w", err))
		return nil
	}
	return
}
