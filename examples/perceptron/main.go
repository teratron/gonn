package main

import (
	"fmt"
	"time"

	"github.com/teratron/gonn/pkg/nn"
)

func main() {
	_ /*n :*/ = nn.New[float32]()

	dataSet := []float32{.27, -.31, -.52, .66, .81, -.13, .2, .49, .11, -.73, .28} // Dataset.
	lenInput := 3                                                                  // Number of input data.
	lenOutput := 2                                                                 // Number of output data.
	lenData := len(dataSet) - lenOutput
	start := time.Now() // Starting the timer.

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
		if sum/num < 1e-6 /*n.GetLossLimit*/ {
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
