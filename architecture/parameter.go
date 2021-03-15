package architecture

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/nn/hopfield"
	"github.com/teratron/gonn/nn/perceptron"
)

// Parameter
type Parameter interface {
	gonn.Parameter
	perceptron.Parameter
	hopfield.Parameter
}
