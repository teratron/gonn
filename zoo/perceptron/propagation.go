package perceptron

import (
	"math"

	"github.com/teratron/gonn/params"
)

// calcNeuron
func (p *perceptron) calcNeuron(input []float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range p.neuron {
		if i > 0 {
			dec = i - 1
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}

		for j, n := range v {
			go func(j int, n *neuron) {
				n.value = 0
				for k, w := range p.Weights[i][j] {
					if k < length {
						if i > 0 {
							n.value += p.neuron[dec][k].value * w
						} else {
							n.value += input[k] * w
						}
					} else {
						n.value += w
					}
				}
				n.value = params.Activation(n.value, p.Activation)
				wait <- true
			}(j, n)
		}

		for range v {
			<-wait
		}
	}
}

// calcLoss calculating the error of the output neuron
func (p *perceptron) calcLoss(target []float64) (loss float64) {
	for i, n := range p.neuron[p.lastLayerIndex] {
		n.miss = target[i] - n.value
		switch p.Loss {
		default:
			fallthrough
		case params.ModeMSE, params.ModeRMSE:
			loss += math.Pow(n.miss, 2)
		case params.ModeARCTAN:
			loss += math.Pow(math.Atan(n.miss), 2)
		}
		n.miss *= params.Derivative(n.value, p.Activation)
	}

	loss /= float64(p.lenOutput)
	if p.Loss == params.ModeRMSE {
		loss = math.Sqrt(loss)
	}
	return
}

// calcMiss calculating the error of neurons in hidden layers
func (p *perceptron) calcMiss() {
	wait := make(chan bool)
	defer close(wait)

	for i := p.lastLayerIndex - 1; i >= 0; i-- {
		inc := i + 1
		for j, n := range p.neuron[i] {
			go func(j int, n *neuron) {
				n.miss = 0
				for k, m := range p.neuron[inc] {
					n.miss += m.miss * p.Weights[inc][k][j]
				}
				n.miss *= params.Derivative(n.value, p.Activation)
				wait <- true
			}(j, n)
		}

		for range p.neuron[i] {
			<-wait
		}
	}
}

// updWeight update weights
func (p *perceptron) updWeight(input []float64) {
	wait := make(chan bool)
	defer close(wait)

	var length, dec int
	for i, v := range p.Weights {
		if i > 0 {
			dec = i - 1
			length = len(p.neuron[dec])
		} else {
			length = p.lenInput
		}
		for j, w := range v {
			go func(i, j, dec, length int, grad float64, w []float64) {
				for k := range w {
					if k < length {
						if i > 0 {
							p.Weights[i][j][k] += p.neuron[dec][k].value * grad
						} else {
							p.Weights[i][j][k] += input[k] * grad
						}
					} else {
						p.Weights[i][j][k] += grad
					}
				}
				wait <- true
			}(i, j, dec, length, p.neuron[i][j].miss*p.Rate, w)
		}
	}

	for _, v := range p.Weights {
		for range v {
			<-wait
		}
	}
}
