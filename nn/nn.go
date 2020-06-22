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
	FloatType	float32

	bias		*Bias
	input		*neuron
	output		*neuron
)

//
type nn struct {
	architecture	NeuralNetwork	// Architecture/type of neural network (configuration)
	isInit			bool			// Neural network initialization flag
	isTrain			bool

	upperRange		FloatType // Range, Bound, Limit, Scope
	lowerRange		FloatType

	//
	language		string
	logging			bool

	//
	neuron			[]neuron
	axon			[]axon
}

//
type neuron struct {
	modeActivation uint8
	value          FloatType
	error          FloatType
	axon           []axon
}

//
type axon struct {
	weight  FloatType         //
	synapse map[string]Vertex // map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}