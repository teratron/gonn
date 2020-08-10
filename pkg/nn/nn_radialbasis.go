// Radial Basis Neural Network - under construction
package nn

/*import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

type radialBasis struct {
	Parameter
	Constructor
}

// Returns a new Radial Basis neural network instance with the default parameters
func (n *nn) RadialBasis() NeuralNetwork {
	n.Architecture = &radialBasis{
		Parameter: n,
	}
	return n
}

// Setter
func (r *radialBasis) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		switch v := args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Radial Basis Neural Network", true)
			log.Printf("\tset: %T %v\n", v, v) // !!!
		}
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Getter
func (r *radialBasis) Get(args ...pkg.Getter) pkg.GetterSetter {
	if len(args) > 0 {
		switch args[0].(type) {
		default:
			pkg.Log("This type of variable is missing for Radial Basis Neural Network", true)
			log.Printf("\tget: %T %v\n", args[0], args[0]) // !!!
			return nil
		}
	} else {
		return r
	}
}

// Initialization
func (r *radialBasis) init(args ...pkg.Setter) bool {
	return true
}*/

// Train
/*func (r *radialBasis) Train(input, target []float64) (loss float64, count int) {
	return
}

// Query
func (r *radialBasis) Query(input []float64) []float64 {
	panic("implement me")
}*/