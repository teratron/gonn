package perceptron

import (
	"fmt"
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

// Verify verifying dataset.
func (nn *NN) Verify(input []float64, target ...[]float64) float64 {
	var err error
	if len(input) > 0 {
		if len(target) > 0 && len(target[0]) > 0 {
			nn.mutex.Lock()
			defer nn.mutex.Unlock()

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

			_ = copy(nn.input, input)
			_ = copy(nn.output, target[0])
			//nn.input = input
			//nn.output = target[0]

			nn.calcNeuron()
			return nn.calcLoss()
		} else {
			err = pkg.ErrNoTarget
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("verify: %v\n", err)
	return -1
}
