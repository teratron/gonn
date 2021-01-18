package nn

// MaxIteration the maximum number of iterations after which training is forcibly terminated
const MaxIteration int = 10e+05

// NeuralNetwork
type NeuralNetwork interface {
	Parameter
	ReadWriter

	// Query
	Query(input []float32) (output []float32)

	// Verify
	Verify(input []float32, target ...[]float32) (loss float32)

	// Train
	Train(input []float32, target ...[]float32) (loss float32, count int)
}
