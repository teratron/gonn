package perceptron

import (
	"sync"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
	"github.com/teratron/gonn/pkg/utils"
)

// NAME of the neural network architecture.
const NAME = "perceptron"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

// NN
type NN struct {
	pkg.Parameter `json:"-"`

	// Neural network architecture name (required field for a config).
	Name string `json:"name"`

	// The neuron bias, false or true (required field for a config).
	Bias bool `json:"bias"`

	// Array of the number of neurons in each hidden layer.
	Hidden []int `json:"hidden,omitempty"`

	// Activation function mode (required field for a config).
	Activation uint8 `json:"activation"`

	// The mode of calculation of the total error.
	Loss uint8 `json:"loss"`

	// Minimum (sufficient) limit of the average of the error during training.
	Limit float64 `json:"limit"`

	// Learning coefficient (greater than 0 and less than or equal to 1).
	Rate pkg.FloatType `json:"rate"`

	// Weight value.
	Weights pkg.Float3Type `json:"weights,omitempty"`

	// Neuron.
	neuron [][]*neuron

	// Settings.
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	isInit         bool
	config         utils.Filer
	mutex          sync.Mutex

	// Transfer data.
	input  []float64
	output []float64
}

type neuron struct {
	value pkg.FloatType
	miss  pkg.FloatType
}

// New return Perceptron neural network.
func New() *NN {
	return &NN{
		Name:       NAME,
		Activation: params.SIGMOID,
		Loss:       params.MSE,
		Limit:      .01,
		Rate:       .3,
	}
}
