package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/network"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	Network network.Network[T] `json:"network,omitempty" toml:"network,omitempty" xml:"network,omitempty"`
	//Activation activation.Type
	//Loss       loss.Type
	//Rate       T
	//Bias       bool
	isInit  bool
	isTrain bool
	isQuery bool
}

func New[T utils.Float]() *NN[T] {
	utils.Logger.Info("Neural network initialized")

	return &NN[T]{
		Network: network.New[T](),
		//Activation: activation.Default,
		//Loss:       loss.DEFAULT,
		//Rate:       0.3,
		//Bias:       false,
		isInit:  false,
		isTrain: false,
		isQuery: false,
	}
}

func (n *NN[T]) Dense(size uint, activationMode activation.Type, bias bool) *NN[T] {
	/*n.hiddenLayers = append(n.hiddenLayers, HiddenLayer{
		Number:     size,
		Activation: activation,
		Bias:       bias, //n.useBias,
	})*/
	n.Network.Hidden.Init(int(size), activationMode, bias)
	//network.NewLayer[T](size, activation, bias)
	return n
}
