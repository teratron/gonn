package hopfield

import (
	"github.com/teratron/gonn"
)

//const HopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ gonn.NeuralNetwork = (*Hopfield)(nil)

// hopfield
type Hopfield struct {
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
	neuron []*hopfieldNeuron

	// Settings
	lenInput int
	isInit   bool
	jsonName string
}

func (h Hopfield) Read(reader gonn.Reader) error {
	panic("implement me")
}

func (h Hopfield) Write(writer ...gonn.Writer) error {
	panic("implement me")
}

// hopfieldNeuron
type hopfieldNeuron struct {
	value float64
}

// Hopfield return
/*func Hopfield() *Hopfield {
	return &Hopfield{
		Name: HopfieldName,
	}
}
*/
