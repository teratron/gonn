package nn

import "github.com/zigenzoog/gonn/pkg"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

// NeuralNetwork
type NeuralNetwork interface {
	Parameter
	Constructor
	pkg.Controller
}

// nn
type nn struct {
	// Architecture of neural network
	Architecture `json:"architecture,omitempty" xml:"architecture,omitempty"`

	// State of the neural network
	isInit  bool // Neural network initializing flag
	isTrain bool // Neural network training flag

	json string
}
