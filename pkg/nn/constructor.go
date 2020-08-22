package nn

import "github.com/zigenzoog/gonn/pkg"

//
type Constructor interface {
	// Initializing
	init(int, ...interface{}) bool

	// Querying
	Query(input []float64) (output []float64)

	// Verifying
	Verify(input []float64, target ...[]float64) (loss float64)

	// Training
	Train(input []float64, target ...[]float64) (loss float64, count int)

	// Copying
	Copy(pkg.Getter)

	// Pasting
	Paste(pkg.Getter) error

	// Adding
	//Add()

	// Deleting
	//Delete()
}