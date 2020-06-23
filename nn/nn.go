//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	Architecture
	GetterSetter

	//Parameter
	//Processor
}

//
type Architecture interface {
	//
	//Perceptron() NeuralNetwork
	Perceptron
	//
	//RadialBasis() NeuralNetwork
	RadialBasis

	//
	//Hopfield() NeuralNetwork
	Hopfield
}

//
type Processor interface {
	// Initializing
	Init()

	// Training
	Train()

	// Querying / forecast / prediction
	Query()

	// Verifying / validation
	Verify()
}

type Vertex interface {
}

type (
	floatType	float32

	bias		*biasType
	input		*neuron
	output		*neuron
)

//
type nn struct {
	architecture	NeuralNetwork	// Architecture/type of neural network (configuration)
	isInit			bool			// Neural network initialization flag
	isTrain			bool

	//
	language		string
	logging			bool

	//
	neuron			[]neuron
	axon			[]axon
}

//
type neuron struct {
	value          floatType
	error          floatType
	axon           []axon
}

//
type axon struct {
	weight  floatType         //
	synapse map[string]Vertex // map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}