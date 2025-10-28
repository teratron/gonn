package main

import (
	"path/filepath"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network from config.
	n := nn.New[float32](filepath.Join("config", "perceptron.json"))

	// Dataset.
	input := []float32{1, 1}
	target := []float32{0}

	// Training dataset.
	_, _ = n.Train(input, target)

	// Writing weights to a file.
	_ = n.WriteWeights("perceptron_weights.json")
}
