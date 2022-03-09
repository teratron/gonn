package perceptron

import (
	"log"
	"math"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
)

// calcNeuron.
/*func (nn *NN) calcNeuron() {
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
				for k, w := range nn.weight[i][j] {
					if k < length {
						if i > 0 {
							n.value += nn.neuron[dec][k].value * w
						} else {
							n.value += pkg.FloatType(nn.input[k]) * w
						}
					} else {
						n.value += w
					}
					num++
				}

				switch nn.Activation {
				case params.LINEAR:
					if num > 0 {
						n.value /= num
					}
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
}*/
func (nn *NN) calcNeuron() {
	var length, dec int
	for i, v := range nn.neuron {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		for j, n := range v {
			var num pkg.FloatType = 0
			n.value = 0
			for k, w := range nn.weight[i][j] {
				if k < length {
					if i > 0 {
						n.value += nn.neuron[dec][k].value * w
					} else {
						n.value += nn.input[k] * w
					}
				} else {
					n.value += w
				}
				num++
			}

			switch nn.ActivationMode {
			case params.LINEAR:
				if num > 0 {
					n.value /= num
				}
			default:
				n.value = params.Activation(n.value, nn.ActivationMode)
			}
		}
	}
}

// calcLoss calculating the error of the output neuron.
func (nn *NN) calcLoss() (loss float64) {
	for i, n := range nn.neuron[nn.lastLayerIndex] {
		n.miss = nn.target[i] - n.value
		switch nn.LossMode {
		default:
			fallthrough
		case params.MSE, params.RMSE:
			loss += math.Pow(float64(n.miss), 2)
		case params.ARCTAN:
			loss += math.Pow(math.Atan(float64(n.miss)), 2)
		case params.AVG:
			loss += math.Abs(float64(n.miss))
		}
	}

	loss /= float64(nn.lenOutput)
	if nn.LossMode == params.RMSE {
		loss = math.Sqrt(loss)
	}

	switch {
	case math.IsNaN(loss):
		log.Panic("calcLoss: loss not-a-number value") // TODO: log.Panic (?)
	case math.IsInf(loss, 0):
		log.Panic("calcLoss: loss is infinity") // TODO: log.Panic (?)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers.
/*func (nn *NN) calcMiss() {
	if nn.lastLayerIndex > 0 {
		wait := make(chan bool)
		defer close(wait)

		for i := nn.lastLayerIndex - 1; i >= 0; i-- {
			inc := i + 1
			for j, n := range nn.neuron[i] {
				go func(j int, n *neuron) {
					n.miss = 0
					for k, m := range nn.neuron[inc] {
						n.miss += m.miss * nn.weight[inc][k][j]
					}
					wait <- true
				}(j, n)
			}

			for range nn.neuron[i] {
				<-wait
			}
		}
	}
}*/
func (nn *NN) calcMiss() {
	if nn.lastLayerIndex > 0 {
		for i := nn.lastLayerIndex - 1; i >= 0; i-- {
			inc := i + 1
			for j, n := range nn.neuron[i] {
				n.miss = 0
				for k, m := range nn.neuron[inc] {
					n.miss += m.miss * nn.weight[inc][k][j]
				}
			}
		}
	}
}

// updateWeight update weights.
/*func (nn *NN) updateWeight() {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range nn.weight {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		for j, w := range v {
			grad := nn.Rate * nn.neuron[i][j].miss * params.Derivative(nn.neuron[i][j].value, nn.ActivationMode)
			go func(i, j, dec, length int, grad pkg.FloatType, w []pkg.FloatType) {
				for k := range w {
					if k < length {
						var value pkg.FloatType
						if i > 0 {
							value = nn.neuron[dec][k].value
						} else {
							value = pkg.FloatType(nn.input[k])
						}

						switch nn.Activation {
						case params.LINEAR:
							if value != 0 {
								nn.weight[i][j][k] += grad / value
							}
						default:
							nn.weight[i][j][k] += grad * value
						}
					} else {
						nn.weight[i][j][k] += grad
					}
				}
				wait <- true
			}(i, j, dec, length, grad, w)
		}
	}

	for _, v := range nn.weight {
		for range v {
			<-wait
		}
	}
}*/
func (nn *NN) updateWeight() {
	var length, dec int
	for i, v := range nn.weight {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		for j, w := range v {
			grad := nn.Rate * nn.neuron[i][j].miss * params.Derivative(nn.neuron[i][j].value, nn.ActivationMode)
			for k := range w {
				if k < length {
					var value pkg.FloatType
					if i > 0 {
						value = nn.neuron[dec][k].value
					} else {
						value = nn.input[k]
					}

					switch nn.ActivationMode {
					case params.LINEAR:
						if value != 0 {
							nn.weight[i][j][k] += grad / value
						}
					default:
						nn.weight[i][j][k] += grad * value
					}
				} else {
					nn.weight[i][j][k] += grad
				}
			}
		}
	}
}

/*func (nn *NN) updateWeight() {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range nn.weight {
		if i > 0 {
			dec = i - 1
			length = len(nn.neuron[dec])
		} else {
			length = nn.lenInput
		}

		update := func(i, j, dec, length int, grad pkg.FloatType, w []pkg.FloatType) {
			for k := range w {
				if k < length {
					var value pkg.FloatType
					if i > 0 {
						value = nn.neuron[dec][k].value
					} else {
						value = pkg.FloatType(nn.input[k])
					}

					switch nn.ActivationMode {
					case params.LINEAR:
						if value != 0 {
							nn.weight[i][j][k] += grad / value
						}
					default:
						nn.weight[i][j][k] += grad * value
					}
				} else {
					nn.weight[i][j][k] += grad
				}
			}
		}

		routine := func(i, j, dec, length int, grad pkg.FloatType, w []pkg.FloatType) {
			update(i, j, dec, length, grad, w)
			wait <- true
		}

		for j, w := range v {
			grad := nn.Rate * nn.neuron[i][j].miss * params.Derivative(nn.neuron[i][j].value, nn.ActivationMode)
			go routine(i, j, dec, length, grad, w)
		}
	}

	for _, v := range nn.weight {
		for range v {
			<-wait
		}
	}
}*/
