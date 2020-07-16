// Calculating
package nn

type Calculator interface {
	calc(...Initer) Getter
}

func (n *nn) calc(args ...Initer) (get Getter) {
	if len(args) > 0 {
		if a, ok := n.Get().(NeuralNetwork); ok {
			for _, v := range args {
				if i, ok := v.(Initer); ok {
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