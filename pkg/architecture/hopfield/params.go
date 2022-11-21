package hopfield

import "github.com/teratron/gonn/pkg"

// GetEnergy.
func (nn *NN) GetEnergy() float64 {
	return nn.Energy
}

// SetEnergy.
func (nn *NN) SetEnergy(energy float64) {
	nn.Energy = energy
}

// GetWeight.
func (nn *NN) GetWeight() pkg.Floater {
	return &nn.Weight
}

// SetWeight.
func (nn *NN) SetWeight(weight pkg.Floater) {
	if w, ok := weight.(pkg.Float2Type); ok {
		nn.Weight = w
	}
}
