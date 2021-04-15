package perceptron

import (
	"fmt"
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

// Query querying dataset
func (nn *NN) Query(input []float64) (output []float64) {
	var err error
	if len(input) > 0 {
		if !nn.isInit {
			err = pkg.ErrInit
			goto ERROR
		} else if nn.lenInput != len(input) {
			err = fmt.Errorf("invalid number of elements in the input data")
			goto ERROR
		}

		nn.calcNeuron(input)
		output = make([]float64, nn.lenOutput)
		for i, n := range nn.neuron[nn.lastLayerIndex] {
			output[i] = float64(n.value)
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	if err != nil {
		log.Println(fmt.Errorf("query: %w", err))
		return nil
	}
	return
}
