package nn

import (
	"log/slog"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	Bias       bool
	Rate       T
	Activation activation.Type
	Loss       loss.Type
	Network    []network.Network[T]

	logger *slog.Logger
	isInit  bool
	isQuery bool
}

func New[T utils.Float]() *NN[T] {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	logger.Info("Neural network initialized")

	return &NN[T]{
		Bias:       true,
		Rate:       T(0.01),
		Activation: activation.RELU,
		Loss:       loss.MSE,
		Network:    []network.Network[T]{},
		logger:     logger,
		isInit:     false,
		isQuery:    false,
	}
}
