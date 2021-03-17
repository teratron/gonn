package hopfield

// Parameter
type Parameter interface {
	NeuronEnergy() float64
	SetNeuronEnergy(float64)
}
