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
	Weights [][]FloatType `json:"weights" xml:"weights"`

	// Matrix
	neuron []FloatType

	// State of the neural network
	isInit  bool
	isTrain bool

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
func (h *hopfield) NeuronEnergy() float32 {
	return float32(h.Energy)
}

// Set
func (h *hopfield) Set(args ...Setter) {
	fmt.Print(args)
}

// Get
func (h *hopfield) Get(args ...Getter) GetSetter {
	fmt.Print(args)
	return h
}

// Read
func (h *hopfield) Read(reader Reader) {
	fmt.Print(reader)
}

// Write
func (h *hopfield) Write(writer ...Writer) {
	fmt.Print(writer)
}
