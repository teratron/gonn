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
		//init(...Setter) bool
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
	init(...Setter) bool
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

type (
	intType			int
	floatType		float32
	FloatType		[]float64
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
	//index			int
	value			floatType
	axon			[]*axon
}

//
type axon struct {
	//index			int
	weight			floatType			//
	synapse			map[string]Setter	// map["bias"]Vertex, map["input"]Vertex, map["output"]Vertex
}