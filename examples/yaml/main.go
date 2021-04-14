package main

import (
	"path/filepath"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New(filepath.Join("config", "perceptron.yml"))

	// Training dataset
	input := []float64{1, 1}
	target := []float64{0}

	// Training
	_, _ = n.Train(input, target)

	// Writing the neural network configuration to a file
	_ = n.WriteWeight("perceptron_weights.yml")
}
