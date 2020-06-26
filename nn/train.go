package nn

//
func (n *nn) Train(data ...[]float64) (loss float64, count int) {
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		n.isInit = n.Init(data[0], data[1])
	}
	/*forwardPass(input)
	totalLoss(target)
	n.backwardPass()*/
	return
}

//
func (n *nn) backwardPass() {
}