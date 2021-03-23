package gonn

// NeuralNetwork neural network interface.
type NeuralNetwork interface {
	Parameter
	//ReadWriter
	//DecodeEncoder

	// Init
	Init(data ...interface{}) error

	// Query
	Query(input []float64) (output []float64)

	// Verify
	Verify(input []float64, target ...[]float64) (loss float64)

	// Train
	Train(input []float64, target ...[]float64) (loss float64, count int)
}
