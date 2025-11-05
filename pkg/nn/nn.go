package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	Network    network.Network[T] `json:"network,omitempty" yaml:"network,omitempty" toml:"network,omitempty" xml:"network,omitempty"`
	Activation activation.Type
	Loss       loss.Type
	Rate       T
	Bias       bool
	isInit     bool
	isTrain    bool
	isQuery    bool
}

func New[T utils.Float]() *NN[T] {
	utils.Logger.Info("Neural network initialized")

	return &NN[T]{
		Network:    network.New[T](),
		Activation: activation.DEFAULT,
		Loss:       loss.DEFAULT,
		Rate:       0.3,
		Bias:       false,
		isInit:     false,
		isTrain:    false,
		isQuery:    false,
	}
}
