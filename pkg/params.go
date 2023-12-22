package pkg

import "github.com/teratron/gonn/pkg/nn"

// Parameter.
type Parameter interface {
	GetLengthInput() int
	GetLengthOutput() int

	GetWeights() nn.Floater
	SetWeights(nn.Floater)

	GetHiddenLayer() []uint
	SetHiddenLayer(...uint)

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
