// Calculating
package nn

type Calculator interface {
	calc(...Initer) Getter
	//Initer
}

func (n *nn) calc(args ...Initer) (get Getter) {
	if len(args) == 0 {
		Log("Empty calc()", false)
	} else {
		if a, ok := getArchitecture(n); ok {
			for _, v := range args {
				if i, ok := v.(Initer); ok {
					g := a.calc(i)
					if g != nil { get = g }
				}
			}
		}
	}
	return
}