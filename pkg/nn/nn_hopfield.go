// Package nn - under construction
package nn

import "fmt"

const hopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

type hopfield struct {
	NeuralNetwork `json:"-" xml:"-"`
	//Parameter     `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	Energy FloatType `json:"energy" xml:"energy"`

	// Weights values
	Weights Float2Type `json:"weights" xml:"weights"`

	// Matrix
	neuron []FloatType

	// State of the neural network
	isInit bool // Initializing flag

	// Config
	jsonName string
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: hopfieldName,
	}
}

// NeuronEnergy
func (h *hopfield) NeuronEnergy() float64 {
	return float64(h.Energy)
}

// SetNeuronEnergy
func (h *hopfield) SetNeuronEnergy(energy float64) {
	h.Energy = FloatType(energy)
}

// Read
func (h *hopfield) Read(reader Reader) {
	fmt.Print(reader)
}

// Write
func (h *hopfield) Write(writer ...Writer) {
	fmt.Print(writer)
}
