package nn

func init() {
}

//
func New() NeuralNetwork {
	return NeuralNetwork{
		Specifier: &FeedForward{},
		isInit:    false,
		Rate:      DefaultRate,
		LossMode:  ModeMSE,
		LossLimit: .0001,

		UpperRange: 1,
		LowerRange: 0,
	}
}
