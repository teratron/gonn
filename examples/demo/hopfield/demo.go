package main

import "github.com/zigenzoog/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// for Hopfield neural network
	n := nn.New(nn.Hopfield())
	println("nn.New(nn.Hopfield():", n)
}
