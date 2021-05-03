package perceptron

import (
	"fmt"
	"log"
	"math"

	"github.com/teratron/gonn/pkg"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration = 10e+05

var GetMaxIteration = getMaxIteration

func getMaxIteration() int {
	return MaxIteration
}

// Train training dataset.
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
				count++
				nn.calcNeuron(&input)
				switch loss = nn.calcLoss(&target[0]); {
				case loss < nn.Limit:
					return
				case math.IsNaN(loss), math.IsInf(loss, 0):
					log.Panic("train: not optimal neural network parameters")
				}
				nn.calcMiss()
				nn.updWeight(&input)
			}
			//fmt.Printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b")
			//fmt.Printf("%d %.5f",count, loss)
			return
		} else {
			err = pkg.ErrNoTarget
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("train: %v\n", err)
	return -1, 0
}
