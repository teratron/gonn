package params

// CheckHiddenLayer
func CheckHiddenLayer(layer []int) []int {
	if len(layer) > 0 {
		return layer
	}
	return []int{0}
}
