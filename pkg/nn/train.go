package nn

const MaxIteration uint = 1_000_000

// Train trains the neural network.
func (n *NN[T]) Train(input, target *[]T) (count uint, loss T) {
	return MaxIteration, loss
}
