package nn

// Parameter
type Parameter interface {
	Weight() Floater

	// Perceptron
	HiddenLayer() []uint
	Bias() bool
	ActivationMode() uint8
	LossMode() uint8
	LossLevel() float64
	Rate() float32

	// Hopfield
	Energy() float32
}