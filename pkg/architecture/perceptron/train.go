package perceptron

import (
	"fmt"
	"log"

	"github.com/teratron/gonn/pkg"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration int = 1e+09

var GetMaxIteration = getMaxIteration

func getMaxIteration() int {
	return MaxIteration
}

// Train training dataset.
func (nn *NN) Train(input []float64, target ...[]float64) (count int, loss float64) {
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

			nn.input = pkg.ToFloat1Type(input)
			nn.target = pkg.ToFloat1Type(target[0])

			return nn.train()
		} else {
			err = pkg.ErrNoTarget
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("perceptron.NN.Train: %v\n", err)
	return 0, -1
}

// AndTrain the training dataset after the query.
func (nn *NN) AndTrain(target []float64) (count int, loss float64) {
	var err error
	if len(target) > 0 {
		nn.mutex.Lock()
		defer nn.mutex.Unlock()

		if !nn.isInit {
			err = pkg.ErrInit
			goto ERROR
		} else if nn.lenOutput != len(target) {
			err = fmt.Errorf("invalid number of elements in the target data")
			goto ERROR
		}

		nn.target = pkg.ToFloat1Type(target)

		return nn.train()
	} else {
		err = pkg.ErrNoTarget
	}

ERROR:
	log.Printf("perceptron.NN.AndTrain: %v\n", err)
	return 0, -1
}

func (nn *NN) train() (count int, loss float64) {
	minLoss := 1.
	minCount := 0
	for count < GetMaxIteration() {
		if !nn.isQuery {
			nn.calcNeurons()
		} else {
			nn.isQuery = false
		}
		count++

		if loss = nn.calcLoss(); loss < minLoss {
			minLoss = loss
			minCount = count
			nn.weights = pkg.DeepCopy(nn.Weights)
			if loss < nn.LossLimit {
				nn.Weights = pkg.DeepCopy(nn.weights)
				return minCount, minLoss
			}
		}
		nn.calcMiss()
		nn.updateWeights()
	}

	if minCount > 0 {
		nn.Weights = pkg.DeepCopy(nn.weights)
	}
	return minCount, minLoss
}
