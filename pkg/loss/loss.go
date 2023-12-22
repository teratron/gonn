package loss

// The mode of calculation of the total error.
const (
	MSE    uint8 = iota // MSE - Mean Squared Error.
	RMSE                // RMSE - Root Mean Squared Error.
	ARCTAN              // ARCTAN - Arctan Error.
	AVG                 // AVG - Average Error.
)

// CheckLossMode.
func CheckLossMode(mode uint8) uint8 {
	if mode > AVG {
		return MSE
	}
	return mode
}
