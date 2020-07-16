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
	/*if a, ok := n.Architecture.(*perceptron); ok {
		//a.neuron
	}*/

	return nil//[]float64(n.Architecture.(*perceptron).neuron[len(p.neuron) - 1])
}

//
func forwardPass(input []float64) []float64 {
	/*
	m.CalcNeuron()
	return m.Layer[m.Index].Neuron*/
	return []float64{0}
}