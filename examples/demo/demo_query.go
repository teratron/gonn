package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("config/perceptron.json"))

	// Input dataset
	input := []float64{.27, .31, .52}

	// Neural network query
	output := n.Query(input)
	fmt.Println(output)
}
