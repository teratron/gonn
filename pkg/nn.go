package pkg

// NeuralNetwork interface.
type NeuralNetwork interface {
	Parameter

	// Init.
	Init(data ...interface{})

	// Query.
	Query(input []float64) (output []float64)

	// Verify.
	Verify(input []float64, target ...[]float64) (loss float64)

	// Train.
	Train(input []float64, target ...[]float64) (count int, loss float64)

	// AndTrain.
	AndTrain(target []float64) (count int, loss float64)

	// WriteConfig writes the configuration and weights.
	WriteConfig(filename ...string) error

	// WriteWeights writes weights.
	WriteWeights(filename string) error
}
