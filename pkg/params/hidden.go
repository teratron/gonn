package params

// CheckHiddenLayer.
func CheckHiddenLayer(layer []uint) []uint {
	if len(layer) > 0 {
		return layer
	}
	return []uint{0}
}
