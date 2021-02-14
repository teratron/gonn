package main

import (
	"path/filepath"

	"github.com/zigenzoog/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New(nn.Perceptron())
	n := nn.New()

	// The neuron bias, false or true
	n.SetNeuronBias(true)

	// Array of the number of neurons in each hidden layer
	n.SetHiddenLayer(5, 3)

	// Activation function mode
	n.SetActivationMode(nn.ModeTANH)

	// The mode of calculation of the total error
	n.SetLossMode(nn.ModeMSE)

	// Minimum (sufficient) limit of the average of the error during training
	n.SetLossLimit(.001)

	// Learning coefficient, from 0 to 1
	n.SetLearningRate(nn.DefaultRate)

	// Training dataset
	dataSet := []float64{.27, .31, .52, .66, .81, .13, .2, .49, .11, .73, .28}
	lenInput := 3  // Number of input data
	lenOutput := 2 // Number of output data

	// Training
	var buff nn.Floater
	minLoss := 1.
	limit := len(dataSet) - lenOutput
	for epoch := 1; epoch <= 1000; epoch++ {
		for i := lenInput; i <= limit; i++ {
			_, _ = n.Train(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
		}

		// Verifying
		sum := 0.
		num := 0
		for i := lenInput; i <= limit; i++ {
			sum += n.Verify(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
			num++
		}

		// Average error for the entire epoch
		sum /= float64(num)

		// Weights are copied to the buffer at the minimum average error
		if sum < minLoss {
			buff = n.Weight()
			minLoss = sum
		}

		// Exiting the cycle of learning epochs, when the minimum error level is reached
		if sum <= n.LossLimit() {
			break
		}
	}

	// Returning weights for further recording from the buffer
	n.SetWeight(buff)

	// Writing the neural network configuration to a file
	_ = n.Write(nn.JSON(filepath.Join(".", "perceptron.json")))
}
