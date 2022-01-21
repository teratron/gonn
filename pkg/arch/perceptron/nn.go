package perceptron

import (
	"sync"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/params"
	"github.com/teratron/gonn/pkg/utils"
)

// Name of the neural network architecture.
const Name = "perceptron"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

// NN.
type NN struct {
	pkg.Parameter `json:"-"`

	// Neural network architecture name (required field for a config).
	Name string `json:"name"`

	// The neuron bias, false or true (required field for a config).
	Bias bool `json:"bias"`

	// Array of the number of neurons in each hidden layer.
	HiddenLayer []uint `json:"hiddenLayer,omitempty"`

	// Activation function mode (required field for a config).
	ActivationMode uint8 `json:"activationMode"`

	// The mode of calculation of the total error.
	LossMode uint8 `json:"lossMode"`

	// Minimum (sufficient) limit of the average of the error during training.
	LossLimit float64 `json:"lossLimit"`

	// Learning coefficient (greater than 0 and less than or equal to 1).
	Rate pkg.FloatType `json:"rate"`

	// Weight value.
	Weight pkg.Float3Type `json:"weight,omitempty"`

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
	input  pkg.Float1Type
	output pkg.Float1Type
	weight pkg.Float3Type
}

type neuron struct {
	value pkg.FloatType
	miss  pkg.FloatType
}

// New return Perceptron neural network.
func New() *NN {
	return &NN{
		Name:           Name,
		ActivationMode: params.SIGMOID,
		LossMode:       params.MSE,
		LossLimit:      .01,
		Rate:           .3,
	}
}
