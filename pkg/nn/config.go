package nn

import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

// HiddenLayer represents a hidden layer configuration
type HiddenLayer struct {
	Number     uint
	Activation activation.Type
	Bias       bool
}

// OutputLayer represents an output layer configuration
type OutputLayer struct {
	Number     uint
	Activation activation.Type
	Loss       loss.Type
	Bias       bool
}

// SetHiddenLayers configures the hidden layers of the neural network
func (n *NN[T]) SetHiddenLayers(layers ...HiddenLayer) *NN[T] {
	return n
}

// SetOutputLayer configures the output layer of the neural network
func (n *NN[T]) SetOutputLayer(
	number uint, activation activation.Type, loss loss.Type, bias bool,
) *NN[T] {
	return n
}

// SetRate sets the learning rate for the neural network
func (n *NN[T]) SetRate(value float64) *NN[T] {
	if value < 0.0 {
		utils.Logger.Warn("Rate cannot be negative", "value", value)
		return n
	}
	n.Rate = T(value)
	return n
}
