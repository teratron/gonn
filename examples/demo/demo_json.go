package main

import "github.com/zigenzoog/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("config/perceptron.json"))

	// Training dataset
	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	// Training
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file
	n.Write(nn.JSON())
}