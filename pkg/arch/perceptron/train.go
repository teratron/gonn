package perceptron

import (
	"fmt"
	"log"
	"math"

	"github.com/teratron/gonn/pkg"
)

// MaxIteration the maximum number of iterations after which training is forcibly terminated.
const MaxIteration int = 1e+06

var GetMaxIteration = getMaxIteration

func getMaxIteration() int {
	return MaxIteration
}

// MinLossLimit minimum (sufficient) limit of the average of the error during training.
const MinLossLimit = 1e-15

var GetMinLossLimit = getMinLossLimit

func getMinLossLimit() float64 {
	return MinLossLimit
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

			_ = copy(nn.input, input)
			_ = copy(nn.output, target[0])

			if nn.Weight[0][0][0] != 0 {
				_ = copy(nn.weight, nn.Weight)
			}
			//fmt.Println(nn.weight)

			minLoss := 1.
			minCount := 0
			for count < GetMaxIteration() {
				count++
				nn.calcNeuron()
				loss = nn.calcLoss()

				switch {
				case math.IsNaN(loss), math.IsInf(loss, 0):
					log.Panic("train: not optimal neural network parameters")
				case loss < minLoss:
					minLoss = loss
					minCount = count
					_ = copy(nn.Weight, nn.weight)
					if loss < nn.LossLimit || loss < GetMinLossLimit() {
						fmt.Println(count, "---MinLossLimit", minCount, minLoss)
						return minCount, minLoss
					}
				}
				nn.calcMiss()
				nn.updateWeight()
			}
			fmt.Println("+++++", minCount, minLoss)
			return minCount, minLoss
		} else {
			err = pkg.ErrNoTarget
		}
	} else {
		err = pkg.ErrNoInput
	}

ERROR:
	log.Printf("train: %v\n", err)
	return 0, -1
}

// TODO: // AndTrain training dataset.
/*func (nn *NN) AndTrain(target ...[]float64) (loss float64, count int) {
	_ = copy(nn.output, target[0])

	for count < GetMaxIteration() {
		count++
		loss = nn.calcLoss()
		switch {
		case loss < GetMinLossLimit():
			return
		case math.IsNaN(loss), math.IsInf(loss, 0):
			log.Panic("train: not optimal neural network parameters")
		}
		nn.calcMiss()
		nn.updateWeight()
	}
	return
}*/
