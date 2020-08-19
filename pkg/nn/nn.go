package nn

import "github.com/zigenzoog/gonn/pkg"

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*NN)(nil)

//
type NeuralNetwork interface {
	//
	Architecture

	//
	Parameter

	//
	Constructor

	//
	pkg.GetterSetter

	//
	pkg.ReaderWriter
}

type Architecture interface {
	//perceptron() NeuralNetwork
	//hopfield() NeuralNetwork
}

//
type NN struct {
	Architecture			`json:"architecture,omitempty" xml:"architecture,omitempty"`	// Architecture of neural network
	Parameter				`json:"-" xml:"-"`

	IsInit       bool		`json:"-" xml:"-"`				// Neural network initializing flag
	IsTrain      bool		`json:"isTrain" xml:"isTrain"`	// Neural network training flag

	json		string
	xml			xmlType
	csv			csvType
	db			dbType
}

/*type NN struct {
	*nn
}*/