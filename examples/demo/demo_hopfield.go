package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// for Hopfield neural network
	n := nn.New().Hopfield()

	fmt.Println(n)
}