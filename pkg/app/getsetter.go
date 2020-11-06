package app

import "github.com/zigenzoog/gonn/pkg"

// Set
func (a *app) Set(args ...pkg.Setter) {
}

// Get
func (a *app) Get(args ...pkg.Getter) pkg.GetSetter {
	return a
}
