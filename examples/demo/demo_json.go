package main

import (
	"fmt"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	n := nn.New(nn.JSON("config/perceptron.json"))

	fmt.Println("nn.New(JSON(\"file\")):", n)

	//
	n.Write(
		nn.JSON(),
		nn.XML())
}