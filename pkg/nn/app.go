//
package nn

type app struct {
	language	langType
	logging		modeLogType
}

// New returns a new neural network instance with the default parameters
func App() *app {
	a := &app{
		language:	"en",
		logging:	1,
	}
	return a
}