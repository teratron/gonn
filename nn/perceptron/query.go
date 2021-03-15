package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// Query querying dataset
func (p *perceptron) Query(input []float64) (output []float64) {
	var err error
	if len(input) > 0 {
		if !p.isInit {
			err = gonn.ErrInit
			goto ERROR
		} else if p.lenInput != len(input) {
			err = fmt.Errorf("invalid number of elements in the input data")
			goto ERROR
		}
		p.calcNeuron(input)
		output = make([]float64, p.lenOutput)
		for i, n := range p.neuron[p.lastLayerIndex] {
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
