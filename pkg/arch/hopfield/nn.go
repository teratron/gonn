package hopfield

import (
	"sync"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// NAME of the neural network architecture.
const NAME = "hopfield"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

type NN struct {
	pkg.NeuralNetwork `json:"-"`
	//gonn.Parameter     `json:"-"`

	// Neural network architecture name (required field for config).
	Name string `json:"name"`

	// Energy.
	Energy float64 `json:"energy"`

	// Weights values.
	Weights pkg.Float2Type `json:"weights,omitempty"`

	// Neuron.
	neuron []*neuron

	// Settings.
	lenInput int
	isInit   bool
	config   utils.Filer
	mutex    sync.Mutex
}

type neuron struct {
	value float64
}

// New return Hopfield neural network.
func New() *NN {
	return &NN{
		Name: NAME,
	}
}
