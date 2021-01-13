package nn

// checkHiddenLayer
func checkHiddenLayer(layer []int) []int {
	if len(layer) > 0 {
		return layer
	} else {
		return []int{0}
	}
}
