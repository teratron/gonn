//
package nn

// Declare conformity with Application interface
var _ Application = (*app)(nil)

type Application interface {
	//
	GetterSetter
}
type app struct {
	language	langType
	logging		modeLogType
}

// New returns a new application instance with the default parameters
func App() *app {
	a := &app{
		language:	"en",
		logging:	1,
	}
	return a
}