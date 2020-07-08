package nn

const MaxIteration uint32 = 10e+05	// The maximum number of iterations after which training is forcibly terminated

//
// data[0] - input data
// data[1] - target data
// ... - any data
func (n *nn) Train(data ...[]float64) (loss float64, count int) {
//func (n *nn) Train(input, target []float64) (loss float64, count int) {
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		n.isInit = n.init(FloatType(data[0]), FloatType(data[1]))
	}
	/*forwardPass(input)
	totalLoss(target)
	n.backwardPass()*/
	return
}

//
func (n *nn) backwardPass() {
}