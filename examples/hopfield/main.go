package main

import (
	"fmt"

	"github.com/teratron/gonn/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// for Hopfield neural network
	n := nn.New(nn.Hopfield())
	fmt.Println("nn.New(nn.Hopfield():", n)
}
