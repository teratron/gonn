package perceptron

import (
	"fmt"
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

// Verify verifying dataset
func (nn *NN) Verify(input []float64, target ...[]float64) (loss float64) {
	var err error
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			if !nn.isInit {
				nn.Init(len(input), len(target[0]))
			} else {
				if nn.lenInput != len(input) {
					err = fmt.Errorf("invalid number of elements in the input data")
					goto ERROR
				}
				if nn.lenOutput != len(target[0]) {
					err = fmt.Errorf("invalid number of elements in the target data")
					goto ERROR
				}
			}

			nn.calcNeuron(input)
			loss = nn.calcLoss(target[0])
		} else {
			err = pkg.ErrNoTarget
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	if err != nil {
		log.Println(fmt.Errorf("verify: %w", err))
		return -1
	}
	return
}
