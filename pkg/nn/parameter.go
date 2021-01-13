package nn

// Parameter
type Parameter interface {
	name() string
	setName(string)

	stateInit() bool
	setStateInit(bool)

	nameJSON() string
	setNameJSON(string)

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

	LearningRate() float32
	SetLearningRate(float32)

	NeuronEnergy() float64
	SetNeuronEnergy(float64)

	Weight() Floater
	SetWeight(Floater)
}
