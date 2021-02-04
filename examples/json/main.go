package main

import (
	"fmt"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("./perceptron.json"))
	//fmt.Println(n)
	// Training dataset
	input := []float64{.05, .1}
	target := []float64{.01, .99}

	// Training
	//_, _ = n.Train(input, target)

	// Neural network query
	//output := n.Query(input)
	//fmt.Println(output)

	fmt.Println("loss: ", n.Verify(input, target))

	// Writing the neural network configuration to a file
	//_ = n.Write(nn.JSON("./perceptron_2.json"))
}
