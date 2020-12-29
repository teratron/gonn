package nn

// MaxIteration the maximum number of iterations after which training is forcibly terminated
const MaxIteration int = 10e+05

// NeuralNetwork
type NeuralNetwork interface {
	Parameter
	ReadWriter

	// Error
	Error() string

	// Query
	Query(input []float64) (output []float64)

	// Verify
	Verify(input []float64, target ...[]float64) (loss float64)

	// Train
	Train(input []float64, target ...[]float64) (loss float64, count int)
}
