package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// Neural Network
	n := nn.New(nn.JSON("file"))

	fmt.Println("nn.New(JSON(\"file\")):", n)
}