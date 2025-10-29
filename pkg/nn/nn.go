package nn

import (
	"log"
	"os"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/utils"
)

type NN[T utils.Float] struct {
	*log.Logger
}

func New[T utils.Float]() *NN[T] {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	logger.Println("Neural network initialized")

	return &NN[T]{}
}

type HiddenLayer struct {
	Number     uint
	Activation activation.Type
	Bias       bool
}

func (n *NN[T]) SetHiddenLayers(layers ...HiddenLayer) *NN[T] {
	return n
}

func (n *NN[T]) SetOutputLayer(number uint, activation activation.Type, loss loss.Type, bias bool) *NN[T] {
	return n
}

func (n *NN[T]) SetRate(value float64) *NN[T] {
	if value < 0 {
		log.Println("Rate cannot be negative")
		return n
	}
	return n
}
