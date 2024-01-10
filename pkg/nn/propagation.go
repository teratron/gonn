package nn

import (
	"log"
	"math"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// calcNeurons.
func (nn *NN[T]) calcNeurons() {
	length, dec := nn.lenInput, 0
	for i, v := range nn.neurons {
		if i > 0 {
			dec = i - 1
			length = len(nn.neurons[dec])
		}

		for j, n := range v {
			var num T = 0
			n.value = 0
			for k, w := range nn.Weights[i][j] {
				if k < length {
					if i > 0 {
						n.value += nn.neurons[dec][k].value * w
					} else {
						n.value += nn.input[k] * w
					}
				} else {
					n.value += w
				}
				num++
			}

			if nn.ActivationMode == activation.LINEAR {
				if num > 0 {
					n.value /= num
				}
				n.value = 1 // TODO:
			} else {
				n.value = activation.Activation(n.value, nn.ActivationMode)
			}

			if i == nn.lastLayerIndex {
				nn.output[j] = n.value
			}
		}
	}
}

// calcLoss calculating the error of the output neuron.
func (nn *NN[T]) calcLoss() (los T) {
	for i, n := range nn.neurons[nn.lastLayerIndex] {
		n.miss = nn.target[i] - n.value
		switch nn.LossMode {
		default:
			fallthrough
		case loss.MSE, loss.RMSE:
			los += utils.Pow(n.miss, 2)
		case loss.ARCTAN:
			los += utils.Pow(math.Atan(float64(n.miss)), 2)
		case loss.AVG:
			los += math.Abs(float64(n.miss))
		}
	}

	// TODO: math.Copysign()

	switch { // TODO:
	case utils.IsNaN(los):
		log.Panic("1:perceptron.NN.calcLoss: loss not-a-number value") // TODO: log.Panic (?)
	case utils.IsInf(los, 0):
		log.Panic("1:perceptron.NN.calcLoss: loss is infinity") // TODO: log.Panic (?)
	}

	los /= T(nn.lenOutput)
	if nn.LossMode == loss.RMSE {
		los = math.Sqrt(los)
	}

	switch {
	case math.IsNaN(los):
		log.Panic("2:perceptron.NN.calcLoss: loss not-a-number value") // TODO: log.Panic (?)
	case math.IsInf(los, 0):
		log.Panic("2:perceptron.NN.calcLoss: loss is infinity") // TODO: log.Panic (?)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers.
func (nn *NN) calcMiss() {
	//if nn.lastLayerIndex > 0 {
	// for i := nn.lastLayerIndex - 1; i >= 0; i-- {
	for i := nn.prevLayerIndex; i >= 0; i-- {
		inc := i + 1
		for j, n := range nn.neurons[i] {
			n.miss = 0
			for k, m := range nn.neurons[inc] {
				n.miss += m.miss * nn.Weights[inc][k][j]
			}
		}
	}
	//}
}

// updateWeights update weights.
func (nn *NN) updateWeights() {
	length, dec := nn.lenInput, 0
	for i, v := range nn.Weights {
		if i > 0 {
			dec = i - 1
			length = len(nn.neurons[dec])
		}

		for j, w := range v {
			grad := nn.Rate * nn.neurons[i][j].miss * activation.Derivative(nn.neurons[i][j].value, nn.ActivationMode)
			for k := range w {
				if k < length {
					var value nn.FloatType
					if i > 0 {
						value = nn.neurons[dec][k].value
					} else {
						value = nn.input[k]
					}

					if nn.ActivationMode == activation.LINEAR {
						if value != 0 {
							nn.Weights[i][j][k] += grad / value
						}
					} else {
						nn.Weights[i][j][k] += grad * value
					}
				} else {
					nn.Weights[i][j][k] += grad
				}
			}
		}
	}
}
