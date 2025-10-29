package nn

import (
	"log"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
}

func New[T utils.Float]() *NN[T] {
	log.Println("Neural network initialized")

	return &NN[T]{}
}

type HiddenLayer struct {
	Number     uint
	Activation activation.Type
	Bias       bool
}

func (n *NN[T]) SetHiddenLayers(layers ...HiddenLayer) *NN[T] {
	return &NN[T]{}
}

func (n *NN[T]) SetOutputLayer(number uint, activation activation.Type, loss loss.Type, bias bool) *NN[T] {
	return &NN[T]{}
}
