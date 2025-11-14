package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	Network network.Network[T] `json:"network,omitempty" toml:"network,omitempty" xml:"network,omitempty"`

	isInit  bool
	isTrain bool
	isQuery bool
}

func New[T utils.Float]() *NN[T] {
	utils.Logger.Info("Neural network initialized")

	return &NN[T]{
		Network: network.New[T](),
		isInit:  false,
		isTrain: false,
		isQuery: false,
	}
}

func (n *NN[T]) Input(size uint) *NN[T] {
	//n.Network.Input.Init(int(size))
	//network.NewLayer[T](size, activation.None, false)
	return n
}

func (n *NN[T]) Output(size uint, activationMode activation.Type, bias bool) *NN[T] {
	n.Network.Output.Init(int(size), activationMode, bias)
	//network.NewLayer[T](size, activation, bias)
	return n
}

func (n *NN[T]) Dense(size uint, activationMode activation.Type, bias bool) *NN[T] {
	n.Network.Hidden.Init(int(size), activationMode, bias)
	//network.NewLayer[T](size, activation, bias)
	return n
}
