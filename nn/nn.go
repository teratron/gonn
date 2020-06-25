//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	Architecture
	GetterSetter
	Processor
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
	//Init()

	// Training
	Train([]float64, []float64) (float64, int)

	// Querying / forecast / prediction
	Query([]float64) []float64

	// Verifying / validation
	//Verify()
}

type Vertex interface {
}

type (
	floatType		float32

	bias			*biasType
	input			*neuron
	output			*neuron
)

//
type nn struct {
	architecture	NeuralNetwork	// Architecture/type of neural network (configuration)
	isInit			bool			// Neural network initialization flag
	isTrain			bool

	//
	neuron			[]neuron
	axon			[]axon

	//
	language		string
	logging			bool
}

//
type neuron struct {
	architecture	Architecture	// feature
	value			floatType
	axon			[]axon
}

//
type axon struct {
	weight			floatType			//
	synapse			map[string]Vertex	// map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}