//
package nn

const (
	MaxIteration uint32	= 10e+05 // The maximum number of iterations after which training is forcibly terminated

	ModeMSE      uint8	= 0      // Mean Squared Error
	ModeRMSE     uint8	= 1      // Root Mean Squared Error
	ModeARCTAN   uint8	= 2      // Arctan
)

func init() {
	Log("Start", false)
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	return &nn{
		architecture:	&perceptron{},
		isInit:			false,
		isTrain:		false,

		upperRange:		1,
		lowerRange:		0,

		language:		"en",
		logging: 		true,
	}
}

//
func (n *nn) Init(input, target []Float) bool {
	return true
}