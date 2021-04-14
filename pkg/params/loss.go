package params

// The mode of calculation of the total error.
const (
	ModeMSE    uint8 = iota // Mean Squared Error.
	ModeRMSE                // Root Mean Squared Error.
	ModeARCTAN              // Arctan.
)

// CheckLossMode
func CheckLossMode(mode uint8) uint8 {
	if mode > ModeARCTAN {
		return ModeMSE
	}
	return mode
}
