package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration = 10e+05 // 1000000

var GetMaxIteration = getMaxIteration

func getMaxIteration() int {
	return MaxIteration
}

// Train training dataset
func (nn *NN) Train(input []float64, target ...[]float64) (loss float64, count int) {
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

			for count < GetMaxIteration() {
				nn.calcNeuron(input)
				if loss = nn.calcLoss(target[0]); loss <= nn.Limit {
					break
				}
				nn.calcMiss()
				nn.updWeight(input)
				count++
			}
		} else {
			err = gonn.ErrNoTarget
		}
	} else {
		err = gonn.ErrNoInput
	}

ERROR:
	if err != nil {
		log.Println(fmt.Errorf("train: %w", err))
		return -1, 0
	}
	return
}
