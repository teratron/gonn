package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	dataSet := []float32{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28} // Dataset.
	lenInput := 3                                                                  // Number of input data.
	lenOutput := 2                                                                 // Number of output data.
	lenData := len(dataSet) - lenOutput

	n, err := nn.NewBuilder[float32]().
		Input(uint(lenInput)).
		Dense(5, activation.SIGMOID, true).
		Output(uint(lenOutput), activation.SOFTMAX, true).
		WithLoss(loss.ARCTAN).
		WithLearningRate(0.3).
		Compile()
	if err != nil {
		fmt.Printf("Compile error: %v\n", err)
		return
	}

	// Set properties.
	//n.SetHiddenLayers(
	//	// neurons, activation, bias
	//	nn.HiddenLayer{Number: 3, Activation: activation.SIGMOID, Bias: true},  // 1st hidden layer.
	//	nn.HiddenLayer{Number: 5, Activation: activation.ReLU, Bias: true},     // 2nd hidden layer.
	//	nn.HiddenLayer{Number: 3, Activation: activation.SIGMOID, Bias: false}, // 3rd hidden layer.
	//).SetOutputLayer(
	//	// neurons, activation, loss, bias
	//	uint(lenOutput), activation.SIGMOID, loss.ARCTAN, false,
	//).SetRate(0.3)

	// Starting the timer.
	start := time.Now()

	// Training.
	for epoch := 1; epoch <= 100_000; epoch++ {
		for i := lenInput; i <= lenData; i++ {
			_, _ = n.Train(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
		}

		// Verifying.
		sum, num := 0.0, 0.0
		for i := lenInput; i <= lenData; i++ {
			//sum += n.Verify(dataSet[i-lenInput:i], dataSet[i:i+lenOutput])
			num++
		}

		// Average error for the entire epoch.
		// Exiting the cycle of learning epochs, when the minimum error level is reached.
		if num > 0.0 && sum/num < 1e-6 /*n.GetLossLimit*/ {
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
