package hopfield

import "github.com/teratron/gonn"

// Parameter
type Parameter interface {
	gonn.Parameter

	NeuronEnergy() float64
	SetNeuronEnergy(float64)
}
