package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	// New returns a new neural network
	// instance with the default parameters,
	// same n := nn.New("perceptron").
	n := nn.New()

	// The neuron bias, false or true.
	n.SetBias(true)

	// Array of the number of neurons in each hidden layer.
	n.SetHiddenLayer(5, 3)

	// ActivationMode function mode.
	n.SetActivationMode(nn.TANH)

	// The mode of calculation of the total error.
	n.SetLossMode(nn.MSE)

	// Minimum (sufficient) limit of the average of the error during training.
	lossLimit := 1e-6
	n.SetLossLimit(lossLimit)

	// Learning coefficient (greater than 0 and less than or equal to 1).
	n.SetRate(.3)

	// Dataset.
	dataSet := []float64{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28}
	lenInput := 3  // Number of input data.
	lenOutput := 2 // Number of output data.

	start := time.Now()

	// Training.
	lenData := len(dataSet) - lenOutput
	for epoch := 1; epoch <= 10000; epoch++ {
		for i := lenInput; i <= lenData; i++ {
			_, _ = n.Train(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
		}

		// Verifying.
		sum, num := 0., 0.
		for i := lenInput; i <= lenData; i++ {
			sum += n.Verify(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
			num++
		}

		// Average error for the entire epoch.
		// Exiting the cycle of learning epochs, when the minimum error level is reached.
		if sum/num < lossLimit {
			break
		}
	}

	fmt.Printf("Elapsed time: %v\n", time.Now().Sub(start))

	// Writing the neural network configuration to a file.
	//_ = n.WriteConfig("perceptron.json")

	// Writing weights to a file.
	//_ = n.WriteWeight("perceptron_weights.json")

	// Check the trained data, the result should be about [-0.13 0.2].
	fmt.Println(n.Query([]float64{-.52, .66, .81}))
}
