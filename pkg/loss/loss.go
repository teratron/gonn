package loss

type Type uint8

// The mode of calculation of the total error.
const (
	MSE    Type = iota // MSE - Mean Squared Error.
	RMSE               // RMSE - Root Mean Squared Error.
	ARCTAN             // ARCTAN - Arctan Error.
	AVG                // AVG - Average Error.
)

// CheckLossMode.
func CheckLossMode(mode Type) Type {
	if mode > AVG {
		return MSE
	}

	return mode
}
