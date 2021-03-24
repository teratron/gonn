package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// Verify verifying dataset
func (p *perceptron) Verify(input []float64, target ...[]float64) (loss float64) {
	var err error
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			if !p.isInit {
				p.initFromNew(len(input), len(target[0]))
			} else {
				if p.lenInput != len(input) {
					err = fmt.Errorf("invalid number of elements in the input data")
					goto ERROR
				}
				if p.lenOutput != len(target[0]) {
					err = fmt.Errorf("invalid number of elements in the target data")
					goto ERROR
				}
			}

			p.calcNeuron(input)
			loss = p.calcLoss(target[0])
		} else {
			err = gonn.ErrNoTarget
		}
	} else {
		err = gonn.ErrNoInput
	}

ERROR:
	if err != nil {
		log.Println(fmt.Errorf("verify: %w", err))
		return -1
	}
	return
}
