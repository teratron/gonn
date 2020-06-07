package nn

//
func New() NeuralNetwork {
	return NeuralNetwork{
		Specifier: &FeedForward{},
		IsInit:    false,
		Rate:      DefaultRate,
		LossMode:  ModeMSE,
		LossLimit: .0001,
	}
}
