package hopfield

import "github.com/teratron/gonn"

// NeuronEnergy
func (h *hopfield) NeuronEnergy() float64 {
	return h.Energy
}

// SetNeuronEnergy
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
