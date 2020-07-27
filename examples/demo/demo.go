package main

import (
	"os"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters
	// same n := nn.New().Perceptron()
	n := nn.New()

	// Set parameters
	n.Set(
		nn.HiddenLayer(3, 2),
		nn.Bias(true),
		nn.Rate(nn.DefaultRate),
		nn.ModeActivation(nn.ModeSIGMOID),
		nn.ModeLoss(nn.ModeMSE),
		nn.LevelLoss(.0001))

	//
	input  := []float64{2.3, 3.1}
	target := []float64{3.6}

	//
	loss, count := n.Train(input, target)

	//
	//fmt.Println(n.Query(input))

	//
	//fmt.Println(n.Verify(input, target))

	//
	file, err := os.Create("report.txt")
	if err != nil {
		os.Exit(1)
	}
	defer func() { _ = file.Close() }()

	//
	n.Write(nn.JSON("perceptron.json"),
		    nn.Report(file, input, loss, count),
		    nn.Report(os.Stdout, input, loss, count))
}