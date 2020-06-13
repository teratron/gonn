//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*NN)(nil)

//
type NeuralNetwork interface {
	Architecture
	Parameter
	//Processor
}

//
type Architecture interface {
	//
	Perceptron() NeuralNetwork

	//
	FeedForward() NeuralNetwork

	//
	//RadialBasis() NeuralNetwork

	//
	//Hopfield() NeuralNetwork
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
	Get() GetterSetter
}

type Setter interface {
	Set(GetterSetter)
}

type Checker interface {
	Check() GetterSetter
}

type Parameter interface {
	Rate() Rate
	GetRate() Rate
	SetRate(Rate)

	Bias() Bias
	GetBias() Bias
	SetBias(Bias)

	GetterSetter
}

type Vertex interface {
}
/*
type Settings interface {
	Bias() Checker
}
*/

type (
	Float		float32
	Rate		float32
	Bias		float32
	Loss		Float
	//inputSet	[]Float
)

//
type NN struct {
	architecture	NeuralNetwork // Architecture/type of neural network (configuration)
	isInit			bool
	rate			Rate
	modeLoss		uint8
	limitLoss		Loss
	upperRange		Float // Range, Bound, Limit, Scope
	lowerRange		Float

	//
	language string

	//
	neuron []neuron
	axon   []axon
}

//
type neuron struct {
	index			uint32
	modeActivation	uint8
	value			Float
	error			Float
	axon			[]axon
}

//
type axon struct {
	index	uint32
	weight	Float
	synapse	map[string]Vertex // map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}
