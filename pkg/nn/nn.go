//
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*NN)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Constructor

	//
	pkg.GetterSetter

	//Settings
}

/*type Settings interface {
	Bias() bool
	HiddenLayer() []uint
}*/

//
type NN struct {
	Architecture		`json:"-"`			// Architecture of neural network

	IsInit  bool		`json:"-"`			// Neural network initializing flag
	IsTrain bool		`json:"isTrain"`	// Neural network training flag

	json	jsonType	`json:"-"`
	xml		xmlType		`json:"-"`
	csv		csvType		`json:"-"`
	db		dbType		`json:"-"`
}

type settings struct {
	Architecture	perceptron
	IsTrain			bool
}
