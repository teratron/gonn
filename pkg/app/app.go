package app

import "github.com/zigenzoog/gonn/pkg"

// Declare conformity with Application interface
var _ Application = (*app)(nil)

// Application
type Application interface {
	pkg.Controller
}

// app
type app struct {
}

// App returns a new application instance with the default parameters
func App() *app {
	return &app{}
}
