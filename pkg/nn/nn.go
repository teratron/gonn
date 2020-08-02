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
	Architecture // Architecture of neural network

	isInit  bool // Neural network initializing flag
	isTrain bool // Neural network training flag

	json	jsonType
	xml		xmlType
	csv		csvType
}
