package main

import "github.com/zigenzoog/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("perceptron.json"))

	// Training dataset
	input := []float64{.27, .31, .52}
	target := []float64{.66, .81}

	// Training
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file
	n.Write(nn.JSON())
}
