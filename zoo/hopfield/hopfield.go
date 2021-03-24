package hopfield

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/utils"
)

// Name of the neural network architecture.
const Name = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ gonn.NeuralNetwork = (*hopfield)(nil)

type hopfield struct {
	gonn.NeuralNetwork `json:"-" yaml:"-"`
	//gonn.Parameter     `json:"-" yaml:"-"`

	// Neural network architecture name
	Name string `json:"name" yaml:"name"`

	// Energy
	Energy float64 `json:"energy" yaml:"energy"`

	// Weights values
	Weights gonn.Float2Type `json:"weights,omitempty" yaml:"weights,omitempty"`

	// Neuron
	neuron []*neuron

	// Settings
	lenInput int
	isInit   bool
	config   utils.Filer
}

type neuron struct {
	value float64
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: Name,
	}
}
