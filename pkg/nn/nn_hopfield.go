package nn

import "fmt"

const hopfieldName = "hopfield"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*hopfield)(nil)

// hopfield
type hopfield struct {
	NeuralNetwork `json:"-" xml:"-"`
	//Parameter     `json:"-" xml:"-"`

	// Neural network architecture name
	Name string `json:"name" xml:"name"`

	// Energy
	Energy float32 `json:"energy" xml:"energy"`

	// Weights values
	Weights Float2Type `json:"weights" xml:"weights"`

	// Neuron
	neuron []*hopfieldNeuron

	// Settings
	lenInput int
	isInit   bool
	jsonName string
}

// hopfieldNeuron
type hopfieldNeuron struct {
	value float32
}

// Hopfield return
func Hopfield() *hopfield {
	return &hopfield{
		Name: hopfieldName,
	}
}

func (h *hopfield) name() string {
	return h.Name
}

func (h *hopfield) setName(name string) {
	h.Name = name
}

func (h *hopfield) stateInit() bool {
	return h.isInit
}

func (h *hopfield) setStateInit(state bool) {
	h.isInit = state
}

func (h *hopfield) nameJSON() string {
	return h.jsonName
}

func (h *hopfield) setNameJSON(name string) {
	h.jsonName = name
}

// NeuronEnergy
func (h *hopfield) NeuronEnergy() float64 {
	return float64(h.Energy)
}

// SetNeuronEnergy
func (h *hopfield) SetNeuronEnergy(energy float64) {
	h.Energy = float32(energy)
}

// Weight
func (h *hopfield) Weight() Floater {
	return &h.Weights
}

// SetWeight
func (h *hopfield) SetWeight(weight Floater) {
	if w, ok := weight.(Float2Type); ok {
		h.Weights = w
	}
}

// Read
func (h *hopfield) Read(reader Reader) {
	fmt.Print(reader)
}

// Write
func (h *hopfield) Write(writer ...Writer) {
	fmt.Print(writer)
}
