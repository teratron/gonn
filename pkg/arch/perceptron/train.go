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

			if nn.Weight[0][0][0] != 0 {
				nn.weight = nn.Weight
			}

			nn.input = pkg.ToFloat1Type(input)
			nn.target = pkg.ToFloat1Type(target[0])

			minLoss := 1.
			minCount := 0
			for count < GetMaxIteration() {
				count++
				nn.calcNeuron()
				loss = nn.calcLoss()

				switch {
				case math.IsNaN(loss):
					log.Panic("train: loss not-a-number value")
				case math.IsInf(loss, 0):
					log.Panic("train: loss is infinity")
				case loss < minLoss:
					minLoss = loss
					minCount = count
					nn.Weight = nn.weight
					if loss < nn.LossLimit {
						return minCount, minLoss
					}
				}
				nn.calcMiss()
				nn.updateWeight()
			}
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

// AndTrain training dataset.
func (nn *NN) AndTrain(target []float64) (count int, loss float64) {
	if nn.Weight[0][0][0] != 0 {
		nn.weight = nn.Weight
	}

	/*for i, n := range nn.neuron[nn.lastLayerIndex] {
		n.value = pkg.FloatType(output[i])
	}*/
	nn.target = pkg.ToFloat1Type(target)

	loss = nn.calcLoss()
	nn.calcMiss()
	nn.updateWeight()

	count = 1
	minLoss := 1.
	minCount := 0
	for count < GetMaxIteration() {
		count++
		nn.calcNeuron()
		loss = nn.calcLoss()
		//fmt.Println(count, loss)

		switch {
		case math.IsNaN(loss):
			log.Panic("and train: loss not-a-number value")
		case math.IsInf(loss, 0):
			log.Panic("and train: loss is infinity")
		case loss < minLoss:
			minLoss = loss
			minCount = count
			nn.Weight = nn.weight
			//fmt.Println("-----", minCount, minLoss)
			if loss < nn.LossLimit {
				fmt.Println("-----", "/", count, loss)
				return minCount, minLoss
			}
		}
		nn.calcMiss()
		nn.updateWeight()
	}
	fmt.Println("+++++", minCount, minLoss, "/", count, loss)
	return minCount, minLoss
}
