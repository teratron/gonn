package pkg

// Parameter.
type Parameter interface {
	Weight() Floater
	SetWeight(Floater)

	HiddenLayer() []int
	SetHiddenLayer(...int)

	NeuronBias() bool
	SetNeuronBias(bool)

	ActivationMode() uint8
	SetActivationMode(uint8)

	LossMode() uint8
	SetLossMode(uint8)

	LossLimit() float64
	SetLossLimit(float64)

	LearningRate() float64
	SetLearningRate(float64)

	NeuronEnergy() float64
	SetNeuronEnergy(float64)
}
