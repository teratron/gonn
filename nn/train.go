package nn

const MaxIteration uint32 = 10e+05	// The maximum number of iterations after which training is forcibly terminated

//
func (n *nn) Train(data ...[]float64) (loss float64, count int) {
//func (n *nn) Train(input, target []float64) (loss float64, count int) {
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		n.isInit = n.init(FloatType(data[0]), FloatType(data[1]))
		//n.isInit = n.init(FloatType(input), FloatType(target))
		if !n.isInit {
			Log("Error initialization", false) // !!!
			return 0, 0
		}
	}

	count = 1
	n.calc()
	/*for count <= int(MaxIteration) {
		//
		if loss = n.Loss(data[1]); loss <= float64(n.architecture.(*perceptron).levelLoss) || loss <= float64(MinLevelLoss) {
			break
		}
		//n.CalcError()
		//n.UpdateWeight()
		count++
	}*/
	/*forwardPass(input)
	totalLoss(target)
	n.backwardPass()*/
	return
}

//
func (n *nn) backwardPass() {
}