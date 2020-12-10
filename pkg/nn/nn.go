package nn

// MaxIteration the maximum number of iterations after which training is forcibly terminated
const MaxIteration int = 10e+05

// NeuralNetwork
type NeuralNetwork interface {
	Parameter
	ReadWriter

	// Querying
	Query(input []float64) (output []float64)

	// Verifying
	Verify(input []float64, target ...[]float64) (loss float64)

	// Training
	Train(input []float64, target ...[]float64) (loss float64, count int)
}
