package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

// Declare conformity with NeuralNetwork interface
var _ NeuralNetwork = (*NN)(nil)

//
type NeuralNetwork interface {
	//
	//Architecture

	//
	Parameter

	//
	Constructor

	//
	pkg.GetSetter

	//
	pkg.CopyPaster

	//
	pkg.ReadWriter
}

//
type NN struct {
	// Architecture of neural network
	Architecture			`json:"architecture,omitempty" xml:"architecture,omitempty"`

	// State of the neural network
	IsInit       bool		`json:"-" xml:"-"`				// Neural network initializing flag
	IsTrain      bool		`json:"isTrain" xml:"isTrain"`	// Neural network training flag

	json		string
	xml			string
	csv			string
	db			dbType
}

/*type NN struct {
	*nn
}*/