package perceptron

import (
	"github.com/teratron/gonn/pkg"
	cfg "github.com/teratron/gonn/pkg/zoo/configurator"
)

type NN struct {
	//pkg.Parameter `json:"-" yaml:"-"`
	pkg.NeuralNetwork `json:"-" yaml:"-"`

	// Neural network architecture name (required field for a config).
	Name string `json:"name" yaml:"name"`

	neuron [][]*neuron
	//axon   [][][]*axon
	//*weight

	lastIndexLayer int
	lenInput       int
	lenOutput      int
}

type neuron struct {
	cfg.Neuroner
	activation uint8
	miss       pkg.FloatType // Miss (error)
}
