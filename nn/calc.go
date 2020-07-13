// Calculating
package nn

type Calculator interface {
	calc(...Initer)
	Initer
}

func (n *nn) calc(args ...Initer) {
	if v, ok := getArchitecture(n); ok {
		v.calc(args...)
	}

	if len(args) == 0 {
		Log("Empty calc()", false)
	} else {
		for _, v := range args {
			if s, ok := v.(Initer); ok {
				s.calc(args...)
			}
		}
	}
}