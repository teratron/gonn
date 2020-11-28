// Package nn - under construction
package nn

import (
	"fmt"

	"github.com/teratron/gonn/pkg"
)

const hopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

type hopfield struct {
	NeuralNetwork `json:"-" xml:"-"`
	//Parameter     `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	Energy  FloatType  `json:"energy" xml:"energy"`
	Weights Float2Type `json:"weights" xml:"weights"`

	// Matrix
	//neuron []*neuron
	//axon   [][]*axon
	//*weight

	// State of the neural network
	isInit   bool
	isTrain  bool
	jsonName string
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: hopfieldName,
	}
}

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
