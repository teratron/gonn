//
package nn

const (
	DefaultRate  Rate	= .3     // Default rate
	MinLimitLoss Loss	= 10e-33 // The minimum value of the error limit at which training is forcibly terminated
	MaxIteration uint32	= 10e+05 // The maximum number of iterations after which training is forcibly terminated
	ModeMSE      uint8	= 0      // Mean Squared Error
	ModeRMSE     uint8	= 1      // Root Mean Squared Error
	ModeARCTAN   uint8	= 2      // Arctan
)

func init() {
	// log
}

// New returns a new neural network instance with the default parameters
func New() NeuralNetwork {
	return &NN{
		architecture:	&feedForward{},
		isInit:			false,
		rate:			DefaultRate,
		modeLoss:		ModeMSE,
		limitLoss:		.0001,
		upperRange:		1,
		lowerRange:		0,
	}
}

func Arguments(param ...GetterSetter) /*Setter*/ {
	/*for _, p := range param {
		if b, ok := p.(Bias); ok {
			fmt.Println(b + 1)
		}
		//fmt.Printf("%T %v\n", p, p)
	}*/
	//fmt.Printf("%T %v\n", param, param)
	//return param
}
