package layers

// CheckLayers.
func CheckLayers(layers []uint) []uint {
	if layers != nil && len(layers) > 0 {
		return layers
	}

	return []uint{0}
}
