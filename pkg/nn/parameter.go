package nn

import "github.com/teratron/gonn/pkg"

// Parameter
type Parameter interface {
	name() string
	setName(string)

	stateInit() bool
	setStateInit(bool)

	json() string
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

	Weight() pkg.Floater
}
