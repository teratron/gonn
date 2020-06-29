//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	Architecture
		//Perceptron() NeuralNetwork
		//RadialBasis() NeuralNetwork
		//Hopfield() NeuralNetwork
	GetterSetter
		//Get(...Setter) Getter
		//Set(...Setter)
	Processor
		//init(...[]float64)
		//Query([]float64) []float64
		//Loss([]float64) float64
		//Train(...[]float64) (float64, int)
		//Copy([]float64) []float64
}

//
type Architecture interface {
	//
	Perceptron() NeuralNetwork
	//Perceptron
	//
	RadialBasis() NeuralNetwork
	//RadialBasis
	//
	Hopfield() NeuralNetwork
	//Hopfield
}

//
type Processor interface {
	// Initializing
	init(...[]float64) bool
	//Init([]float64, []float64) bool

	// Querying / forecast / prediction
	Query([]float64) []float64

	//
	Loss([]float64) float64

	// Training
	Train(...[]float64) (float64, int)
	//Train([]float64, []float64) (float64, int)

	//
	//Copy([]float64) []float64

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
	isTrain			bool			//

	//
	neuron			[]*neuron
	axon			[]*axon
	lastNeuron		int				// Index of the last neuron of the neural network
	lastAxon		int				// Index of the last axon of the neural network

	//
	language		langType
	logging			logType
}

//
type neuron struct {
	Architecture	// feature
	value			floatType
	//axon			[]axon
}

//
type axon struct {
	weight			floatType			//
	synapse			map[string]Vertex	// map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}