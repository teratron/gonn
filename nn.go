package gonn

// NeuralNetwork
type NeuralNetwork interface {
	Parameter
	ReadWriter

	// Query
	Query(input []float64) (output []float64)

	// Verify
	Verify(input []float64, target ...[]float64) (loss float64)

	// Train
	Train(input []float64, target ...[]float64) (loss float64, count int)
}
