package main

import (
	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New()

	//
	n.Read(nn.XML("perceptron.xml"))

	//
	n.Write(nn.XML())
}
