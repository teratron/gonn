package perceptron

import (
	"github.com/teratron/gonn"
	"github.com/teratron/gonn/params"
	"github.com/teratron/gonn/utils"
)

// Name of the neural network architecture.
const Name = "perceptron"

// Declare conformity with NeuralNetwork interface.
var _ gonn.NeuralNetwork = (*NN)(nil)

type NN struct {
	gonn.Parameter `json:"-" yaml:"-"`

	// Neural network architecture name.
	Name string `json:"name" yaml:"name"`

	// The neuron bias, false or true.
	Bias bool `json:"bias" yaml:"bias"`

	// Array of the number of neurons in each hidden layer.
	Hidden []int `json:"hidden,omitempty" yaml:"hidden"`

	// Activation function mode.
	Activation uint8 `json:"activation" yaml:"activation"`

	// The mode of calculation of the total error.
	Loss uint8 `json:"loss" yaml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training.
	Limit float64 `json:"limit" yaml:"limit"`

	// Learning coefficient, from 0 to 1.
	Rate float64 `json:"rate" yaml:"rate"`

	// Weight value.
	Weights gonn.Float3Type `json:"weights,omitempty" yaml:"weights"`

	//*Config

	// Neuron
	neuron [][]*neuron

	// Settings
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	isInit         bool
	config         utils.Filer
}

/*type Config struct {
	gonn.Parameter `json:"-" yaml:"-"`

	// Neural network architecture name.
	Name string `json:"name" yaml:"name"`

	// The neuron bias, false or true.
	Bias bool `json:"bias" yaml:"bias"`

	// Array of the number of neurons in each hidden layer.
	Hidden []int `json:"hidden,omitempty" yaml:"hidden"`

	// Activation function mode.
	Activation uint8 `json:"activation" yaml:"activation"`

	// The mode of calculation of the total error.
	Loss uint8 `json:"loss" yaml:"loss"`

	// Minimum (sufficient) limit of the average of the error during training.
	Limit float64 `json:"limit" yaml:"limit"`

	// Learning coefficient, from 0 to 1.
	Rate float64 `json:"rate" yaml:"rate"`

	// Weight value.
	Weights gonn.Float3Type `json:"weights,omitempty" yaml:"weights"`
}*/

type neuron struct {
	value float64
	miss  float64
}

// New return Perceptron neural network.
func New() *NN {
	return &NN{

		Name:       Name,
		Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .1,
		Rate:       params.DefaultRate,

		/*Config: &Config{
			Name:       Name,
			Activation: params.ModeSIGMOID,
			Loss:       params.ModeMSE,
			Limit:      .1,
			Rate:       params.DefaultRate,
		},*/
	}
}