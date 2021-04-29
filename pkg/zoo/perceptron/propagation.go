package perceptron

import (
	"math"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

// calcNeuron.
func (nn *NN) calcNeuron(input *[]float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range nn.neuron {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		for j, n := range v {
			go func(j int, n *neuron) {
				var num pkg.FloatType = 0
				n.value = 0
				for k, w := range nn.Weights[i][j] {
					if k < length {
						if i > 0 {
							n.value += nn.neuron[dec][k].value * w
						} else {
							n.value += pkg.FloatType((*input)[k] /*nn.input[k]*/) * w
						}
					} else {
						n.value += w
					}
					num++
				}

				switch nn.Activation {
				case params.ModeLINEAR:
					n.value /= num
				default:
					n.value = pkg.FloatType(params.Activation(float64(n.value), nn.Activation))
				}
				wait <- true
			}(j, n)
		}

		for range v {
			<-wait
		}
	}
}

// calcLoss calculating the error of the output neuron.
func (nn *NN) calcLoss(target *[]float64) (loss float64) {
	for i, n := range nn.neuron[nn.lastLayerIndex] {
		n.miss = pkg.FloatType((*target)[i] /*nn.output[i]*/) - n.value
		switch nn.Loss {
		default:
			fallthrough
		case params.ModeMSE, params.ModeRMSE:
			loss += math.Pow(float64(n.miss), 2)
		case params.ModeARCTAN:
			loss += math.Pow(math.Atan(float64(n.miss)), 2)
		case params.ModeAVG:
			loss += math.Abs(float64(n.miss))
		}
		n.miss *= pkg.FloatType(params.Derivative(float64(n.value), nn.Activation))
	}

	loss /= float64(nn.lenOutput)
	if nn.Loss == params.ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers.
func (nn *NN) calcMiss() {
	if nn.lastLayerIndex > 0 {
		wait := make(chan bool)
		defer close(wait)

		for i := nn.lastLayerIndex - 1; i >= 0; i-- {
			inc := i + 1
			for j, n := range nn.neuron[i] {
				go func(j int, n *neuron) {
					n.miss = 0
					for k, m := range nn.neuron[inc] {
						n.miss += m.miss * nn.Weights[inc][k][j]
					}
					n.miss *= pkg.FloatType(params.Derivative(float64(n.value), nn.Activation))
					wait <- true
				}(j, n)
			}

			for range nn.neuron[i] {
				<-wait
			}
		}
	}
}

// updWeight update weights.
func (nn *NN) updWeight(input *[]float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range nn.Weights {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		for j, w := range v {
			go func(i, j, dec, length int, grad pkg.FloatType, w []pkg.FloatType) {
				for k := range w {
					if k < length {
						var value pkg.FloatType
						if i > 0 {
							value = nn.neuron[dec][k].value
						} else {
							value = pkg.FloatType((*input)[k] /*nn.input[k]*/)
						}

						switch nn.Activation {
						case params.ModeLINEAR, params.ModeSIGMOID:
							if value > 0 {
								nn.Weights[i][j][k] += grad / value
							}
						default:
							nn.Weights[i][j][k] += grad * value
						}
					} else {
						nn.Weights[i][j][k] += grad
					}
				}
				wait <- true
			}(i, j, dec, length, nn.neuron[i][j].miss*nn.Rate, w)
		}
	}

	for _, v := range nn.Weights {
		for range v {
			<-wait
		}
	}
}
