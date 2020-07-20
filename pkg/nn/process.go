//
package nn

type Processor interface {
	// Initializing
	init(...[]float64) bool

	// Querying / forecast / prediction
	Query(input []float64) (output []float64)

	//
	//Loss(target []float64) (loss float64)

	// Training
	Train(...[]float64) (loss float64, count uint)

	// Copying
	//Copy([]float64) []float64

	// Verifying / validation
	//Verify()
}
