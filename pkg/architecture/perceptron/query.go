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

		nn.input = pkg.ToFloat1Type(input)
		nn.calcNeurons()
		nn.isQuery = true

		return nn.output
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("perceptron.NN.Query: %v\n", err)
	return nil
}
