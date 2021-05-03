package params

// The mode of calculation of the total error.
const (
	MSE    uint8 = iota // Mean Squared Error.
	RMSE                // Root Mean Squared Error.
	ARCTAN              // Arctan.
)

// CheckLossMode.
func CheckLossMode(mode uint8) uint8 {
	if mode > ARCTAN {
		return MSE
	}
	return mode
}
