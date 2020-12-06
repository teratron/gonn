package app

import "github.com/zigenzoog/gonn/pkg"

// Set
func (a *app) Set(...pkg.Setter) {
}

// Get
func (a *app) Get(...pkg.Getter) pkg.GetSetter {
	return a
}
