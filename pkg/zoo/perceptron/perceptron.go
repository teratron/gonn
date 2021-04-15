package perceptron

import (
	"github.com/zigenzoog/gonn/pkg"
	"github.com/zigenzoog/gonn/pkg/params"
	"github.com/zigenzoog/gonn/pkg/utils"
)

// Name of the neural network architecture.
const Name = "perceptron"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

type NN struct {
	pkg.Parameter `json:"-" yaml:"-"`

	// Neural network architecture name.
	Name string `json:"name" yaml:"name"`

	// The neuron bias, false or true.
	Bias bool `json:"bias" yaml:"bias"`

	// Array of the number of neurons in each hidden layer.
	Hidden []int `json:"hidden" yaml:"hidden"`

	// Activation function mode.
	Activation uint8 `json:"activation" yaml:"activation"`

	// The mode of calculation of the total error.
	Loss uint8 `json:"loss" yaml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training.
	Limit float64 `json:"limit" yaml:"limit"`

	// Learning coefficient, from 0 to 1.
	Rate pkg.FloatType `json:"rate" yaml:"rate"`

	// Weight value.
	Weights pkg.Float3Type `json:"weights,omitempty" yaml:"weights,omitempty"`

	// Neuron
	neuron [][]*neuron

	// Settings
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	isInit         bool
	config         utils.Filer
}

type neuron struct {
	value pkg.FloatType
	miss  pkg.FloatType
}

// New return Perceptron neural network.
func New() *NN {
	return &NN{
		Name:       Name,
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .1,
		Rate:       pkg.FloatType(params.DefaultRate),
	}
}
