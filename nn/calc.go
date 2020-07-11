// Calculating
package nn

type Calculator interface {
	calc(...Initer)
}

func (n *nn) calc(args ...Initer) {
	if v, ok := getArchitecture(n); ok {
		v.calc(args...)
	}
}
