package configurator

import (
	"github.com/teratron/gonn/pkg"
)

// Neuroner
type Neuroner interface {
}

// Synapser
type Synapser interface {
}

// Name of the neural network architecture.
const Name = "configurator"

// Declare conformity with NeuralNetwork interface.
var _ pkg.NeuralNetwork = (*NN)(nil)

type NN struct {
	//pkg.Parameter `json:"-" yaml:"-"`
	pkg.NeuralNetwork `json:"-" yaml:"-"`

	// Neural network architecture name (required field for a config).
	Name string `json:"name" yaml:"name"`

	neuron [][]*neuron
	axon   [][][]*axon
	//*weight

	lastIndexLayer int
	lenInput       int
	lenOutput      int
}

type neuron struct {
	value      pkg.FloatType // Neuron value
	miss       pkg.FloatType // Miss (error)
	axon       *[]axon       // All incoming axons
	activation uint8

	specific Neuroner // Specific option of neuron: miss (error) or other

	Synapser
}

type axon struct {
	weight  pkg.FloatType
	synapse map[string]Synapser
}

// New return Perceptron neural network.
func New() *NN {
	return &NN{
		Name: Name,
		/*Activation: params.ModeSIGMOID,
		Loss:       params.ModeMSE,
		Limit:      .01,
		Rate:       pkg.FloatType(params.DefaultRate),*/
	}
}
