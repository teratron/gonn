// Package nn - under construction
package nn

import (
	"fmt"

	//"github.com/teratron/gonn/pkg"
	//"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg"
)

const hopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

type hopfield struct {
	//Architecture `json:"-" xml:"-"`
	//Parameter   `json:"-" xml:"-"`
	Constructor `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	Energy  FloatType  `json:"energy" xml:"energy"`
	Weights Float2Type `json:"weights" xml:"weights"`

	// Configurations
	/*Conf struct {
		Energy floatType  `json:"energy" xml:"energy"`
		Weight float2Type `json:"weights" xml:"weight"`
	} `json:"hopfield,omitempty" xml:"hopfield,omitempty"`*/

	// Matrix
	//neuron []*neuron
	//axon   [][]*axon
	//*weight
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: hopfieldName,
	}
}

// architecture
/*func (h *hopfield) architecture() Architecture {
	return h.Architecture
}

// setArchitecture
func (h *hopfield) setArchitecture(network Architecture) {
	if n, ok := network.(*nn); ok {
		h.Architecture = n
	}
	h.Energy = .001
}*/

// NeuronEnergy
func (h *hopfield) NeuronEnergy() float32 {
	return float32(h.Energy)
}

// Set
func (h *hopfield) Set(args ...pkg.Setter) {
	fmt.Print(args)
}

// Get
func (h *hopfield) Get(args ...pkg.Getter) pkg.GetSetter {
	fmt.Print(args)
	return h
}

// Copy
func (h *hopfield) Copy(copier pkg.Copier) {
	fmt.Print(copier)
}

// Paste
func (h *hopfield) Paste(paster pkg.Paster) {
	fmt.Print(paster)
}

// Read
func (h *hopfield) Read(reader pkg.Reader) {
	fmt.Print(reader)
}

// Write
func (h *hopfield) Write(writer ...pkg.Writer) {
	fmt.Print(writer)
}

// readJSON
func (h *hopfield) readJSON(value interface{}) {
	fmt.Print(value)
}
