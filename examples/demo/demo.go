package main

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network instance with the default parameters
	// same n := nn.New(Perceptron())
	n := nn.New()

	// Set parameters
	n.Set(
		nn.HiddenLayer(5, 3),
		nn.Bias(true),
		nn.ActivationMode(nn.ModeSIGMOID),
		nn.LossMode(nn.ModeMSE),
		nn.LossLevel(.01),
		nn.Rate(nn.DefaultRate))

	// Training dataset
	dataSet   := []float64{.27, .31, .52, .66, .81, .13, .2, .49, .11, .73, .28, .43}
	numInput  := 3	// Number of input data
	numOutput := 2	// Number of output data

	// Training
	minLoss := 1.
	for epoch := 1; epoch <= 100000; epoch++ {
		for i := numInput; i <= len(dataSet) - numOutput; i++ {
			_, _ = n.Train(dataSet[i - numInput:i], dataSet[i:i + numOutput])
		}

		// Verifying
		sum := 0.
		num := 0
		for i := numInput; i <= len(dataSet) - numOutput; i++ {
			loss := n.Verify(dataSet[i - numInput:i], dataSet[i:i + numOutput])
			sum += loss
			num++
		}

		// Average error for the entire epoch
		sum /= float64(num)

		if epoch == 1 || epoch == 10 || epoch % 100 == 0 || epoch == 100000 {
			fmt.Printf("Epoch: %v\tError: %.8f\n", epoch, sum)
		}

		// Weights are copied at the minimum average error
		if sum < minLoss && epoch >= 1000 {
			fmt.Println("----- Epoch:", epoch, "\tmin avg error:", sum)
			n.Copy(nn.Weight())
			minLoss = sum
		}

		// Exiting the cycle of learning epochs, when the minimum error level is reached
		if sum <= n.LossLevel() {
			fmt.Printf("Epoch: %v\tError: %.8f\n", epoch, sum)
			break
		}
	}

	// Saving the neural network configuration to a file
	n.Write(
		nn.JSON("config/perceptron.json"),
		nn.XML("config/perceptron.xml"))
}