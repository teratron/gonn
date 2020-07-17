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
	Processor
		//Query([]float64) []float64
		//Loss([]float64) float64
		//Train(...[]float64) (float64, int)
		//Copy([]float64) []float64
	GetterSetter
		//Get(...Setter) Getter
		//Set(...Setter)
}

//
type Processor interface {
	init(...[]float64) bool

	calc(...GetterSetter) Getter

	// Querying / forecast / prediction
	Query([]float64) []float64

	//
	Loss([]float64) float64
	//loss(FloatType) floatType

	// Training
	Train(...[]float64) (float64, int)
	//Train([]float64, []float64) (float64, int)

	//
	//Copy([]float64) []float64

	// Verifying / validation
	//Verify()
}

type (
	floatType      float32
	//floatArrayType []float64
)

//
type nn struct {
	Architecture			// Architecture/type of neural network (configuration)

	isInit		bool		// Neural network initialization flag
	isQuery		bool		//
	isTrain		bool		//

	language	langType
	logging		modeLogType
}