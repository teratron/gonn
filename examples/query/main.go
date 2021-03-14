package main

import (
	"fmt"
	"path/filepath"

	"github.com/teratron/gonn/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON(filepath.Join("perceptron.json")))

	// Input dataset
	input := []float64{.27, .31, .52}

	// Neural network query
	output := n.Query(input)
	fmt.Println(output)
}
