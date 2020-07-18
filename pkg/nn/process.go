//
package nn

type Processor interface {
	init(...[]float64) bool

	calc(...GetterSetter) Getter

	// Querying / forecast / prediction
	Query([]float64) []float64

	//
	Loss([]float64) float64
	//loss(FloatType) floatType

	// Training
	Train(...[]float64) (float64, int)
	//Train([]float64, []float64) (float64, int)

	//
	//Copy([]float64) []float64

	// Verifying / validation
	//Verify()
}
