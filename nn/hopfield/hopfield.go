package hopfield

import (
	"github.com/teratron/gonn"
)

// Name of the neural network architecture.
const Name = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ gonn.NeuralNetwork = (*hopfield)(nil)

// hopfield
type hopfield struct {
	//nn.NeuralNetwork `json:"-" xml:"-"`
	gonn.NeuralNetwork `json:"-" xml:"-"`
	//Parameter     `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	// Energy
	Energy float64 `json:"energy" xml:"energy"`

	// Weights values
	Weights gonn.Float2Type `json:"weights" xml:"weights"`

	// Neuron
	neuron []*neuron

	// Settings
	lenInput int
	isInit   bool
	jsonName string
}

// neuron
type neuron struct {
	value float64
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: Name,
	}
}

func (h *hopfield) Get() gonn.Architecture {
	return h
}

/*func (h *hopfield) Set(gonn.Architecture) {
}*/
