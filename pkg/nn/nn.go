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
	IsInit  bool `json:"-" xml:"-"`             // Neural network initializing flag
	IsTrain bool `json:"isTrain" xml:"isTrain"` // Neural network training flag

	json string
}
