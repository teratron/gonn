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

// GetWeights.
func (nn *NN) GetWeights() pkg.Floater {
	return &nn.Weights
}

// SetWeights.
func (nn *NN) SetWeights(weight pkg.Floater) {
	if w, ok := weight.(pkg.Float2Type); ok {
		nn.Weights = w
	}
}
