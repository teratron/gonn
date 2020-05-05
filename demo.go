package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/nn"
)

type Layer []Layer

func main() {
	var mx nn.Matrix

	mx.Bias = 0.1
	fmt.Println(mx.Bias)

	a := nn.Activation{Mode: nn.TANH}
	fmt.Println(a, a.Get(.2))

	neu := nn.Neuron{
		X:	1,
		Y:	0,
		Value:	0.1,
		N:	&nn.PrevNeuronLayer{1, .2, .3},
		W:	&nn.PrevWeightLayer{.3, 52},
	}
	fmt.Println(neu, *neu.N, *neu.W)

	mx.Neuron  = make([][]nn.Neuron, 1)
	mx.Neuron[0] = make([]nn.Neuron, 1)

	mx.Neuron[0][0].Value = 0.02

	fmt.Println(mx.Neuron[0][0].Get())
}
