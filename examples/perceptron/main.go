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

	// The neuron bias, false or true.
	n.SetNeuronBias(true)

	// Array of the number of neurons in each hidden layer.
	n.SetHiddenLayer(5, 3)

	// Activation function mode.
	n.SetActivationMode(nn.TANH)

	// The mode of calculation of the total error.
	n.SetLossMode(nn.MSE)

	// Learning coefficient (greater than 0 and less than or equal to 1).
	n.SetLearningRate(.3)

	// Dataset.
	dataSet := []float64{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28}
	lenInput := 3  // Number of input data.
	lenOutput := 2 // Number of output data.

	// Training.
	lossLimit := .0001
	lenData := len(dataSet) - lenOutput
	for epoch := 1; epoch <= 1; /*0000*/ epoch++ {
		for i := lenInput; i <= lenData; i++ {
			fmt.Println(i)
			_, _ = n.Train(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
		}
		//fmt.Println("epoch:", epoch)
		// Verifying.
		sum, num := 0., 0.
		for i := lenInput; i <= lenData; i++ {
			sum += n.Verify(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
			num++
		}

		// Average error for the entire epoch.
		sum /= num

		fmt.Println("                          epoch:", epoch, " ", sum)

		// Exiting the cycle of learning epochs, when the minimum error level is reached.
		if sum < lossLimit {
			break
		}
	}

	// Writing the neural network configuration to a file.
	_ = n.WriteConfig("perceptron.json")

	// Writing weights to a file.
	_ = n.WriteWeight("perceptron_weights.json")

	// Check the trained data, the result should be about [-0.13 0.2].
	fmt.Println(n.Query([]float64{-.52, .66, .81}))
}
