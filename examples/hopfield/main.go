package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// for Hopfield neural network
	n := nn.New("hopfield")
	n.SetNeuronEnergy(.1)
	fmt.Println("nn.New(\"hopfield\"):", n)
}
