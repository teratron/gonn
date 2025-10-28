package nn

import (
	"sync"

	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
	"github.com/teratron/gonn/pkg/rate"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// NN.
type NN[T pkg.Floater] struct {

	// The neurons bias, false or true (required field for a config).
	Bias bool `json:"bias"`

	// Array of the number of neurons in each hidden layer.
	HiddenLayers []uint `json:"hiddenLayers,omitempty"`

	// Activation function mode (required field for a config).
	ActivationMode activation.Type `json:"activationMode"`

	// The mode of calculation of the total error.
	LossMode loss.Type `json:"lossMode"`

	// Minimum (sufficient) limit of the average of the error during training.
	//LossLimit float64 `json:"lossLimit"`
	LossLimit T `json:"lossLimit"`

	// Learning coefficient (greater than 0 and less than or equal to 1).
	//Rate pkg.FloatType `json:"rate"`
	Rate T `json:"rate"`

	// Weights value.
	//Weights pkg.Float3Type `json:"weights,omitempty"`
	Weights [][][]T `json:"weights,omitempty"`

	// Neurons.
	neurons [][]*neuron[T]

	// Settings.
	lenInput       int
	lenOutput      int
	lastLayerIndex int
	prevLayerIndex int
	isInit         bool
	isQuery        bool
	config         utils.Filer
	mutex          sync.Mutex

	// Transfer data.
	weights [][][]T
	input   []T
	target  []T
	output  []T
}

type neuron[T pkg.Floater] struct {
	value T
	miss  T
}

// New return Perceptron neural network.
func perceptron[T pkg.Floater]() *NN[T] {
	return &NN[T]{
		Bias:           false,
		HiddenLayers:   []uint{0},
		ActivationMode: activation.SIGMOID,
		LossMode:       loss.MSE,
		LossLimit:      .001,
		Rate:           rate.DEFAULT,
	}
}
