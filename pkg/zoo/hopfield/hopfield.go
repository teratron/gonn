package hopfield

import (
	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/utils"
)

// Name of the neural network architecture.
const Name = "hopfield"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

type NN struct {
	pkg.NeuralNetwork `json:"-" yaml:"-"`
	//gonn.Parameter     `json:"-" yaml:"-"`

	// Neural network architecture name (required field for config).
	Name string `json:"name" yaml:"name"`

	// Energy.
	Energy float64 `json:"energy" yaml:"energy"`

	// Weights values.
	Weights pkg.Float2Type `json:"weights,omitempty" yaml:"weights,omitempty"`

	// Neuron.
	neuron []*neuron

	// Settings.
	lenInput int
	isInit   bool
	config   utils.Filer
}

type neuron struct {
	value float64
}

// New return Hopfield neural network.
func New() *NN {
	return &NN{
		Name: Name,
	}
}
