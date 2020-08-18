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


	// Initializing
	/*init(int, ...interface{}) bool

	// Querying
	Query(input []float64) (output []float64)

	// Verifying
	Verify(input []float64, target ...[]float64) (loss float64)

	// Training
	Train(input []float64, target ...[]float64) (loss float64, count int)

	// Copying
	//Copy(dst []float64, src []float64) int

	// Adding
	//Add()

	// Deleting
	//Delete()*/
}

type Architecture interface {
	//Perceptron() *perceptron
	perceptron() NeuralNetwork
	hopfield() NeuralNetwork
}

//
type NN struct {
	Architecture			`json:"architecture,omitempty" xml:"architecture,omitempty"`	// Architecture of neural network

	IsInit       bool		`json:"-" xml:"-"`				// Neural network initializing flag
	IsTrain      bool		`json:"isTrain" xml:"isTrain"`	// Neural network training flag

	json		string
	xml			xmlType
	csv			csvType
	db			dbType

	Parameter				`json:"-" xml:"-"`
}

/*type NN struct {
	*nn
}*/