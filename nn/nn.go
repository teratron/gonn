//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*NN)(nil)

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

type GetterSetter interface {
	Getter
	Setter
}

type Getter interface {
	Get(...Getter) Getter
}

type Setter interface {
	Set(...Setter)
}

type Checker interface {
	Check() Checker
}

type Parameter interface {
	Bias() Bias
	GetBias() Bias
	SetBias(Bias)

	Rate() Rate
	GetRate() Rate
	SetRate(Rate)

	GetHidden() []uint16
	SetHidden(...uint16)
}

type Vertex interface {
}

type (
	Float			float32
	Rate			float32
	Bias			float32
	Loss			Float

	//hidden			func(...uint16) hidden
	//hidden			[]uint16

	bias			*Bias
	input			*neuron
	output			*neuron
)

//
type NN struct {
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
