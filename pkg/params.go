package pkg

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// Parameter.
type Parameter interface {
	GetLengthInput() int
	GetLengthOutput() int

	GetWeights() Floater
	SetWeights(Floater)

	GetHiddenLayer() []uint
	SetHiddenLayer(...uint)

	GetBias() bool
	SetBias(bool)

	GetActivationMode() activation.Type
	SetActivationMode(activation.Type)

	GetLossMode() loss.Type
	SetLossMode(loss.Type)

	GetLossLimit() float64
	SetLossLimit(float64)

	GetRate() float64
	SetRate(float64)
}
