//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	Architecture
	Parameter
	GetterSetter
	//Processor
}

//
type Architecture interface {
	//
	Perceptron() NeuralNetwork

	//
	RadialBasis() NeuralNetwork

	//
	Hopfield() NeuralNetwork
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
	Float			float32

	bias			*Bias
	input			*neuron
	output			*neuron
)

//
type nn struct {
	architecture	NeuralNetwork	// Architecture/type of neural network (configuration)
	isInit			bool			// Neural network initialization flag
	isTrain			bool

	upperRange		Float			// Range, Bound, Limit, Scope
	lowerRange		Float

	//
	language		string
	logging			bool

	//
	neuron			[]neuron
	axon			[]axon
}

//
type neuron struct {
	modeActivation	uint8
	value			Float
	error			Float
	axon			[]axon
}

//
type axon struct {
	weight			Float				//
	synapse			map[string]Vertex	// map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}
