package hopfield

import "github.com/zigenzoog/gonn/pkg"

// NeuronEnergy.
func (nn *NN) NeuronEnergy() float64 {
	return nn.Energy
}

// SetNeuronEnergy.
func (nn *NN) SetNeuronEnergy(energy float64) {
	nn.Energy = energy
}

// Weight.
func (nn *NN) Weight() pkg.Floater {
	return &nn.Weights
}

// SetWeight.
func (nn *NN) SetWeight(weight pkg.Floater) {
	if w, ok := weight.(pkg.Float2Type); ok {
		nn.Weights = w
	}
}
