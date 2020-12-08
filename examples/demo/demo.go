package main

import "github.com/teratron/gonn/pkg/nn"

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New(nn.Perceptron())
	n := nn.New()

	// Set parameters:
	// HiddenLayer    - Array of the number of neurons in each hidden layer
	// NeuronBias     - The neuron bias, false or true
	// ActivationMode - Activation function mode
	// LossMode       - The mode of calculation of the total error
	// LossLimit      - Minimum (sufficient) limit of the average of the error during training
	// LearningRate   - Learning coefficient, from 0 to 1
	n.Set(
		nn.HiddenLayer(5, 3),
		nn.NeuronBias(true),
		nn.ActivationMode(nn.ModeSIGMOID),
		nn.LossMode(nn.ModeMSE),
		nn.LossLimit(.01),
		nn.LearningRate(nn.DefaultRate))

	// Training dataset
	dataSet := []float64{.27, .31, .52, .66, .81, .13, .2, .49, .11, .73, .28}
	numInput := 3  // Number of input data
	numOutput := 2 // Number of output data

	// Training
	minLoss := 1.
	for epoch := 1; epoch <= 1000; epoch++ {
		for i := numInput; i <= len(dataSet)-numOutput; i++ {
			_, _ = n.Train(dataSet[i-numInput:i], dataSet[i:i+numOutput])
		}

		// Verifying
		sum := 0.
		num := 0
		for i := numInput; i <= len(dataSet)-numOutput; i++ {
			sum += n.Verify(dataSet[i-numInput:i], dataSet[i:i+numOutput])
			num++
		}

		// Average error for the entire epoch
		sum /= float64(num)

		// Weights are copied to the buffer at the minimum average error
		if sum < minLoss {
			//n.Copy(nn.Weight())
			minLoss = sum
		}

		// Exiting the cycle of learning epochs, when the minimum error level is reached
		if sum <= n.LossLimit() {
			break
		}
	}

	// Returning weights for further recording from the buffer
	//n.Paste(nn.Weight())

	// Writing the neural network configuration to a file
	n.Write(nn.JSON("perceptron.json"))
}
