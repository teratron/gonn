package params

// CheckLayer.
func CheckLayer(layer []uint) []uint {
	if layer != nil && len(layer) > 0 {
		return layer
	}
	return []uint{0}
}
