package nn

import (
	"log/slog"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

var Logger *slog.Logger

func init() {
	Logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}

type NN[T utils.Float] struct {
	Network    network.Network[T]
	Activation activation.Type
	Loss       loss.Type
	Rate       T
	Bias       bool
	//logger     *slog.Logger
	isInit  bool
	isQuery bool
}

func New[T utils.Float]() *NN[T] {
	//logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	Logger.Info("Neural network initialized")

	return &NN[T]{
		Network:    network.New[T](),
		Activation: activation.DEFAULT,
		Loss:       loss.DEFAULT,
		Rate:       0.3,
		Bias:       false,
		//logger:     logger,
		isInit:  false,
		isQuery: false,
	}
}
