package nn

const (
	// ModeMSE - Mean Squared Error
	ModeMSE uint8 = iota

	// ModeRMSE - Root Mean Squared Error
	ModeRMSE

	// ModeARCTAN - Arctan
	ModeARCTAN
)

// checkLossMode
func checkLossMode(mode uint8) uint8 {
	switch {
	case mode < 0 || mode > ModeARCTAN:
		return ModeMSE
	default:
		return mode
	}
}
