package main

import "github.com/teratron/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("./perceptron.json"))

	// Training dataset
	input := []float64{1, 1}
	target := []float64{0}

	// Training
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file
	_ = n.Write(nn.JSON())
}
