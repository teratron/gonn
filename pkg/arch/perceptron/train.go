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
const MinLossLimit = 1e-24

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

			// TODO:
			//nn.input = input
			//nn.output = target[0]
			_ = copy(nn.input, input)
			_ = copy(nn.output, target[0])

			minLoss := 1.
			minCount := 0

			avgLoss := 0.
			sumLoss := 0.
			/*maxLoss := 1.
			//maxLoss2 := [2]float64{1, 1}
			resistance := 1.*/
			var solid uint32
			for count < GetMaxIteration() {
				count++
				nn.calcNeuron()
				loss = nn.calcLoss()

				sumLoss += loss
				avgLoss = sumLoss / float64(count)

				if loss > avgLoss {
					fmt.Println("!!!!!!!!avgLoss")
				}

				if loss < minLoss {
					minLoss = loss
					minCount = count

					if solid >= 3 {
						_ = copy(nn.Weights, nn.weight)
						/*fmt.Printf("\t\t\t%d: %.20f\n", solid, resistance)
						maxLoss = resistance*/

						fmt.Printf("%d: %.30f\t%.30f\n", minCount, minLoss, avgLoss)
						///fmt.Printf("\t\t\t%d\n", solid)
						solid = 0
					}
					/*resistance = minLoss*/
					//fmt.Printf("%d: %.30f\n", count, loss)
				} else {
					/*if loss > maxLoss {
						fmt.Println("---maxLoss")
						return minCount, minLoss
					}
					if loss > resistance {
						resistance = loss
					}*/
					solid++
				}
				//fmt.Printf("\t\t\t%d\n", solid)

				switch {
				case math.IsNaN(loss), math.IsInf(loss, 0):
					log.Panic("train: not optimal neural network parameters")
				case loss < 0 /*GetMinLossLimit()*/ :
					fmt.Println("---MinLossLimit")
					return minCount, minLoss
				}
				nn.calcMiss()
				nn.updWeight()
			}
			fmt.Println("+++++")
			fmt.Printf("%d: %.30f\t%.30f\n", count, loss, avgLoss)
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

// TODO:
// AndTrain training dataset.
/*func (nn *NN) AndTrain(target ...[]float64) (loss float64, count int) {
	_ = copy(nn.output, target[0])

	for count < GetMaxIteration() {
		count++
		switch loss = nn.calcLoss(); {
		case loss < nn.Limit:
			return
		case math.IsNaN(loss), math.IsInf(loss, 0):
			log.Panic("train: not optimal neural network parameters")
		}
		nn.calcMiss()
		nn.updWeight()
	}
	return
}*/
