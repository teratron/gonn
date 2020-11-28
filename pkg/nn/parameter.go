package nn

// Parameter
type Parameter interface {
	name() string
	setName(string)

	stateInit() bool
	setStateInit(bool)

	stateTrain() bool
	setStateTrain(bool)

	nameJSON() string
	setNameJSON(string)

	// Perceptron
	HiddenLayer() []uint
	NeuronBias() bool
	ActivationMode() uint8
	LossMode() uint8
	LossLimit() float64
	LearningRate() float32

	// Hopfield
	NeuronEnergy() float32

	Weight() Floater
}
