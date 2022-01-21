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

	// Weight values.
	Weight pkg.Float2Type `json:"weight,omitempty"`

	// Neuron.
	neuron []*neuron

	// Settings.
	lenInput int
	isInit   bool
	config   utils.Filer
	mutex    sync.Mutex

	// Transfer data.
	input  pkg.Float1Type
	weight pkg.Float2Type
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
