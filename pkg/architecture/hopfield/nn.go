package hopfield

import (
	"sync"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// Name of the neural network architecture.
const Name = "hopfield"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

// NN.
type NN struct {
	pkg.NeuralNetwork `json:"-"`
	//gonn.Parameter     `json:"-"`

	// Neural network architecture name (required field for config).
	Name string `json:"name"`

	// Energy.
	Energy float64 `json:"energy"`

	// Weights values.
	Weights pkg.Float2Type `json:"weights,omitempty"`

	// Neurons.
	neurons []*neuron

	// Settings.
	lenInput int
	isInit   bool
	config   utils.Filer
	mutex    sync.Mutex

	// Transfer data.
	weights pkg.Float2Type
	input   pkg.Float1Type
}

type neuron struct {
	value float64
}

// New return Hopfield neural network.
func New() *NN {
	return &NN{
		Name: Name,
	}
}
