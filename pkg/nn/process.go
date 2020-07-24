//
package nn

type Processor interface {
	// Initializing
	init(input []float64, target ...[]float64) bool

	// Querying
	Query(input []float64) (output []float64)

	// Verifying
	Verify(input []float64, target ...[]float64) (loss float64)

	// Training
	Train(input []float64, target ...[]float64) (loss float64, count int)

	// Copying
	//Copy([]float64) []float64

	// Deleting
	//Delete()
}
