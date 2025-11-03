package nn

import (
	"log"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	*log.Logger

	Bias       bool
	Rate       T
	Activation activation.Type
	Loss       loss.Type
	Network    []network.Network[T]

	isInit  bool
	isQuery bool
}

func New[T utils.Float]() *NN[T] {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	logger.Println("Neural network initialized")

	return &NN[T]{
		Logger:     logger,
		Bias:       true,
		Rate:       T(0.01),
		Activation: activation.RELU,
		Loss:       loss.MSE,
		Network:    []network.Network[T]{},
		isInit:     false,
		isQuery:    false,
	}
}
