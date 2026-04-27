package nn

/*import (
	"github.com/teratron/gonn/pkg/activation"
	"github.com/teratron/gonn/pkg/loss"
)

// Input sets the size of the input layer
func (n *NN[T]) Input(size uint) *NN[T] {
	n.inputSize = size
	n.inputSet = true
	return n
}

// Dense adds a fully connected hidden layer
func (n *NN[T]) Dense(size uint, activation activation.Type, bias bool) *NN[T] {
	n.hiddenLayers = append(n.hiddenLayers, HiddenLayer{
		Number:     size,
		Activation: activation,
		Bias:       bias, //n.useBias,
	})
	return n
}

// Hidden is an alternative name for Dense
func (n *NN[T]) Hidden(size uint, activation activation.Type, bias bool) *NN[T] {
	return n.Dense(size, activation, bias)
}

// Output sets the output layer
func (n *NN[T]) Output(size uint, activation activation.Type) *NN[T] {
	n.outputSize = size
	n.outputActivation = activation
	n.outputSet = true
	return n
}

// WithLoss sets the loss function
func (n *NN[T]) WithLoss(lossType loss.Type) *NN[T] {
	n.lossFunction = lossType
	return n
}

// WithLearningRate sets the learning rate
func (n *NN[T]) WithLearningRate(rate T) *NN[T] {
	n.learningRate = rate
	return n
}

// WithBias enables/disables bias for all layers
func (n *NN[T]) WithBias(use bool) *NN[T] {
	n.useBias = use
	// Apply to all already added layers
	for i := range n.hiddenLayers {
		n.hiddenLayers[i].Bias = use
	}
	return n
}

// WithWeightInit sets the weight initialization method
func (n *NN[T]) WithWeightInit(method string) *NN[T] {
	n.weightInit = method
	return n
}

// Compile completes the configuration and creates the network
func (n *NN[T]) Compile() (*NN[T], error) {
	// Validation
	if !n.inputSet {
		return nil, utils.NewError("input layer not configured")
	}
	if !n.outputSet {
		return nil, utils.NewError("output layer not configured")
	}
	if n.learningRate == 0 {
		n.learningRate = T(0.01) // default value
	}
	if n.lossFunction == "" {
		n.lossFunction = loss.MSE // default value
	}

	// Create the network
	nn := &NN[T]{
		LearningRate: n.learningRate,
		Loss:         n.lossFunction,
	}

	// Initialize the input layer
	nn.Network.SetInputLayer(n.inputSize)

	// Initialize hidden layers
	if len(n.hiddenLayers) > 0 {
		nn.Network.SetHiddenLayers(n.hiddenLayers...)
	}

	// Initialize the output layer
	nn.Network.SetOutputLayer(n.outputSize, n.outputActivation, n.useBias)

	// Build connections between layers
	if err := nn.Network.Build(n.weightInit); err != nil {
		return nil, err
	}

	return nn, nil
}
*/
