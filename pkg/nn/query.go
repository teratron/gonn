package nn

//
func (n *nn) Query(input []float64) []float64 {
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		Log("An uninitialized neural network", true)
		return nil
	}
	///fmt.Println(floatType(input[0]))
	//forwardPass(input)
	if n.calc(Neuron()) == nil {
		n.isQuery = true
	}
	return nil//[]float64(n.Architecture.(*perceptron).neuron[len(p.neuron) - 1])
}