package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New("perceptron").
	n := nn.New()

	// Parameters.
	n.SetHiddenLayer(3, 2)
	n.SetActivationMode(nn.LINEAR)
	n.SetLossLimit(.001)

	// Dataset that doesn't need to be scaled.
	input := []float64{10.6, 5, 200}
	target := []float64{5, 50.3}

	// Training dataset.
	fmt.Println(n.Train(input, target))

	// Check the trained data, the result should be about [5 50.3].
	fmt.Println(n.Query(input))
}
