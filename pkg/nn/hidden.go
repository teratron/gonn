package nn

// checkHiddenLayer
func checkHiddenLayer(layer []int) []int {
	if len(layer) > 0 {
		return layer
	}
	return []int{0}
}
