package parameter

const (
	// ModeMSE - Mean Squared Error
	ModeMSE uint8 = iota

	// ModeRMSE - Root Mean Squared Error
	ModeRMSE

	// ModeARCTAN - Arctan
	ModeARCTAN
)

// CheckLossMode
func CheckLossMode(mode uint8) uint8 {
	if mode > ModeARCTAN {
		return ModeMSE
	}
	return mode
}
