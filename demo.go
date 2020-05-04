package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/nn"
)

type Layer []Layer

func main() {
	var mx nn.Matrix

	mx.Bias = 0.1
	fmt.Println(mx.Bias /*layermx.Layer[0].Size[0].Neuron*/)

	a := nn.Activation{Mode: nn.TANH}
	fmt.Println(a)
	fmt.Println(a.Get(.2))
}
