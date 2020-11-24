package app

import "github.com/teratron/gonn/pkg"

// Declare conformity with Application interface
var _ Application = (*app)(nil)

// Application
type Application interface {
	pkg.Controller
}

type app struct {
}

// App returns a new application instance with the default parameters
func App() *app {
	return &app{}
}
