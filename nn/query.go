package nn

//
func (n *nn) Query(input []float64) []float64 {
	if n.isInit {
		//copy(m.Layer[0].Neuron, input)
	} else {
		Log("An uninitialized neural network", true)
	}
	///fmt.Println(floatType(input[0]))
	//forwardPropagation(FloatType(input))
	return input
}

//
func forwardPropagation(input []float64) []float64 {
	/*
	m.CalcNeuron()
	return m.Layer[m.Index].Neuron*/
	return []float64{0}
}