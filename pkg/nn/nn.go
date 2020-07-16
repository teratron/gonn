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
	Calculator
		//calc(...Initer) Getter
	Initer
		//init(...Setter) bool
	GetterSetter
		//Get(...Setter) Getter
		//Set(...Setter)
	Processor
		//Query([]float64) []float64
		//Loss([]float64) float64
		//Train(...[]float64) (float64, int)
		//Copy([]float64) []float64
}

//
type Processor interface {
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
	floatArrayType []float64
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