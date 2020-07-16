package nn

// Функция вывода результатов нейросети
/*func (p *perceptron) Print(count int, loss floatType) {
	var (
		i int
		t string
	)
	sep := func() {
		fmt.Println("-----------------------")
	}
	l :=  len(p.neuron)
	o := l - 1
	for i = 0; i < l; i++ {
		switch i {
		case 0:
			t = "Input layer"
		case o:
			t = "Output layer"
		default:
			t = "Hidden layer"
		}
		fmt.Printf("%v %s size: %v\n", i, t, len(p.neuron[i]))
		sep()
		fmt.Println("Neurons:\t", p.neuron)
		fmt.Printf("Errors:\t\t %v\n\n", m.Layer[i].Error)
	}
	fmt.Println("Weights:")
	sep()
	for i = 0; i < m.Index; i++ {
		fmt.Println(m.Synapse[i].Weight)
	}
	sep()
	fmt.Println("Number of iteration:\t", count)
	fmt.Println("Total error:\t", loss)
}*/
