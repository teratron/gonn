// Hopfield Neural Network - under construction
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

type hopfield struct {
	Architecture `json:"-" xml:"-"`
	Parameter    `json:"-" xml:"-"`
	Constructor  `json:"-" xml:"-"`

	Conf struct {
		Energy floatType     `json:"energy" xml:"energy"`
		Weight [][]floatType `json:"weights" xml:"weight"`
	} `json:"hopfield,omitempty" xml:"hopfield,omitempty"`

	// Matrix
	neuron []*neuron
	axon   [][]*axon
	*weight
}

func Hopfield() *hopfield {
	return &hopfield{}
}

// Returns a new Hopfield neural network instance with the default parameters
func (n *nn) hopfield() {
	n.Architecture = &hopfield{
		Architecture: n,
	}
}

func (h *hopfield) architecture() Architecture {
	return h.Architecture
}

func (h *hopfield) setArchitecture(network Architecture) {
	if n, ok := network.(*nn); ok {
		h.Architecture = n
	}
	h.Conf.Weight = nil
}

func (h *hopfield) Energy() float32 {
	return float32(h.Conf.Energy)
}

// Set
func (h *hopfield) Set(args ...pkg.Setter) {}

// Get
func (h *hopfield) Get(args ...pkg.Getter) pkg.GetSetter {
	return h
}

// Copy
func (h *hopfield) Copy(copier pkg.Copier) {}

// Paste
func (h *hopfield) Paste(paster pkg.Paster) {}

// Read
func (h *hopfield) Read(reader pkg.Reader) {}

// Write
func (h *hopfield) Write(writer ...pkg.Writer) {}

// readJSON
func (h *hopfield) readJSON(value interface{}) {}

// readXML
func (h *hopfield) readXML(value interface{}) {}
