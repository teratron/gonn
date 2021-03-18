package hopfield

import "github.com/teratron/gonn"

// Declare conformity with Parameter interface
/*var _ Parameter = (*hopfield)(nil)

// Parameter
type Parameter interface {
	NeuronEnergy() float64
	SetNeuronEnergy(float64)
}*/

func (h *hopfield) NameNN() string {
	return h.Name
}

func (h *hopfield) SetNameNN(name string) {
	h.Name = name
}

func (h *hopfield) InitNN() bool {
	return h.isInit
}

func (h *hopfield) SetInitNN(state bool) {
	h.isInit = state
}

func (h *hopfield) NameJSON() string {
	return h.jsonName
}

func (h *hopfield) SetNameJSON(name string) {
	h.jsonName = name
}

func (h *hopfield) NameYAML() string {
	return h.yamlName
}

func (h *hopfield) SetNameYAML(name string) {
	h.yamlName = name
}

// LossLimit
func (h *hopfield) NeuronEnergy() float64 {
	return h.Energy
}

// SetLossLimit
func (h *hopfield) SetNeuronEnergy(energy float64) {
	h.Energy = energy
}

// Weight
func (h *hopfield) Weight() gonn.Floater {
	return &h.Weights
}

// SetWeight
func (h *hopfield) SetWeight(weight gonn.Floater) {
	if w, ok := weight.(gonn.Float2Type); ok {
		h.Weights = w
	}
}
