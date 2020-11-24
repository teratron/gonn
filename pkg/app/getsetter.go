package app

import "github.com/teratron/gonn/pkg"

// Set
func (a *app) Set(args ...pkg.Setter) {
}

// Get
func (a *app) Get(args ...pkg.Getter) pkg.GetSetter {
	return a
}
