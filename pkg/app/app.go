//
package app

import "github.com/zigenzoog/gonn/pkg"

// Declare conformity with Application interface
var _ Application = (*app)(nil)

type Application interface {
	//
	pkg.GetterSetter
}

type app struct {
	language	pkg.LangType
	logging		pkg.LogModeType
}

// New returns a new application instance with the default parameters
func App() *app {
	a := &app{
		language:	"en",
		logging:	1,
	}
	return a
}