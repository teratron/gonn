package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	n := nn.New[float32]()

	dataSet := []float32{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28} // Dataset.
	lenInput := 3                                                                  // Number of input data.
	lenOutput := 2                                                                 // Number of output data.
	lenData := len(dataSet) - int(lenOutput)
	start := time.Now() // Starting the timer.

	// Set properties.
	n.SetHiddenLayers(
		// neurons, activation, bias
		nn.HiddenLayer{Number: 3, Activation: activation.SIGMOID, Bias: true},
		nn.HiddenLayer{Number: 5, Activation: activation.RELU, Bias: true},
		nn.HiddenLayer{Number: 3, Activation: activation.SIGMOID, Bias: false},
	).SetOutputLayer(
		// neurons, activation, loss, bias
		uint(lenOutput), activation.SIGMOID, loss.ARCTAN, false,
	).SetRate(0.3)

	// Training.
	for epoch := 1; epoch <= 100_000; epoch++ {
		for i := lenInput; i <= lenData; i++ {
			//_, _ = n.Train(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
		}

		// Verifying.
		sum, num := 0., 0.
		for i := lenInput; i <= lenData; i++ {
			//sum += n.Verify(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
			num++
		}

		// Average error for the entire epoch.
		// Exiting the cycle of learning epochs, when the minimum error level is reached.
		if num > 0 && sum/num < 1e-6 /*n.GetLossLimit*/ {
			break
		}
	}

	fmt.Printf("Elapsed time: %v\n", time.Since(start))

	// Writing the neural network configuration to a file.
	//_ = n.WriteConfig("perceptron.json")

	// Writing weights to a file.
	//_ = n.WriteWeights("perceptron_weights.json")

	// Check the trained data, the result should be about [-0.13 0.2].
	//fmt.Println("Check:", n.Query([]float32{-.52, .66, .81}))
}
