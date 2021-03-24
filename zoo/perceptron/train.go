package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration = 10e+05

var GetMaxIteration = getMaxIteration

func getMaxIteration() int {
	return MaxIteration
}

// Train training dataset
func (p *perceptron) Train(input []float64, target ...[]float64) (loss float64, count int) {
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

			for count < GetMaxIteration() {
				p.calcNeuron(input)
				if loss = p.calcLoss(target[0]); loss <= p.Limit {
					break
				}
				p.calcMiss()
				p.updWeight(input)
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
