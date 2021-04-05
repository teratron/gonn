package hopfield

import "github.com/teratron/gonn"

// NeuronEnergy
func (nn *NN) NeuronEnergy() float64 {
	return nn.Energy
}

// SetNeuronEnergy
func (nn *NN) SetNeuronEnergy(energy float64) {
	nn.Energy = energy
}

// Weight
func (nn *NN) Weight() gonn.Floater {
	return &nn.Weights
}

// SetWeight
func (nn *NN) SetWeight(weight gonn.Floater) {
	if w, ok := weight.(gonn.Float2Type); ok {
		nn.Weights = w
	}
}
