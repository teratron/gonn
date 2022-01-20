package pkg

// Parameter.
type Parameter interface {
	GetWeight() Floater
	SetWeight(Floater)

	GetHiddenLayer() []int
	SetHiddenLayer(...int)

	GetBias() bool
	SetBias(bool)

	GetActivationMode() uint8
	SetActivationMode(uint8)

	GetLossMode() uint8
	SetLossMode(uint8)

	GetLossLimit() float64
	SetLossLimit(float64)

	GetRate() float64
	SetRate(float64)

	GetEnergy() float64
	SetEnergy(float64)
}
