package main

import (
	"fmt"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New("perceptron").
	n := nn.New("perceptron")

	// Parameters.
	n.SetBias(true)
	n.SetHiddenLayer(3, 2, 5)
	n.SetActivationMode(nn.LINEAR)
	n.SetLossMode(nn.AVG)
	n.SetLossLimit(.0001)

	// Dataset that doesn't need to be scaled.
	input := []float64{10.6, -5, 200}
	target := []float64{5, -50.3}

	// Training dataset.
	count, loss := n.Train(input, target)
	fmt.Println("Train:", count, loss)

	// Check the trained data, the result should be about [5 -50.3].
	fmt.Println("Check:", n.Query(input))
}
