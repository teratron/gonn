// Calculating
package nn

/*
type Processor interface {
	// Calculating
	calc(...GetterSetter) Getter
}

type Calculator interface {
	calc(...Initer) Getter
}

func (n *nn) calc(args ...GetterSetter) (get Getter) {
	if len(args) > 0 {
		if a, ok := n.Get().(NeuralNetwork); ok {
			for _, v := range args {
				if i, ok := v.(GetterSetter); ok {
					g := a.calc(i)
					if g != nil { get = g }
				}
			}
		}
	} else {
		Log("Empty calc()", true)
	}
	return
}

// Calculating
func (p *perceptron) calc(args ...GetterSetter) Getter {
	if len(args) > 0 {
		for _, a := range args {
			switch v := a.(type) {
			case *neuron:
				p.calcNeuron()
			case *axon:
				p.calcAxon()
			case lossType:
				return levelLossType(p.calcLoss(v))
			default:
				Log("This type is missing for Perceptron Neural Network", true) // !!!
				log.Printf("\tcalc: %T %v\n", args[0], args[0]) // !!!
			}
		}
	} else {
		Log("Empty calc()", true)
	}
	return nil
}

*/