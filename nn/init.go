//
package nn

func init() {
}

//
func New() NeuralNetwork {
	return &neuralNetwork{
		NeuralNetwork: &feedForward{},

		isInit:    false,
		rate:      DefaultRate,
		lossMode:  ModeMSE,
		lossLimit: .0001,

		upperRange: 1,
		lowerRange: 0,
	}
}
