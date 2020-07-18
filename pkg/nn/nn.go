//
package nn

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*nn)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Processor

	//
	GetterSetter
}

//
type nn struct {
	Architecture			// Architecture/type of neural network (configuration)

	isInit		bool		// Neural network initialization flag
	isQuery		bool		//
	isTrain		bool		//

	language	langType
	logging		modeLogType
}