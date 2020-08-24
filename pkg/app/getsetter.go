//
package app

import "github.com/zigenzoog/gonn/pkg"

// Set
func (a *app) Set(args ...pkg.Setter) {
	if len(args) > 0 {
		/*for _, v := range args {
			if s, ok := v.(Setter); ok {
				s.Set(n)
			}
		}*/
	} else {
		pkg.Log("Empty Set()", true) // !!!
	}
}

// Get
func (a *app) Get(args ...pkg.Getter) pkg.GetSetter {
	if len(args) > 0 {
		/*for _, v := range args {
			if g, ok := v.(Getter); ok {
				return g.Get(n)
			}
		}*/
	} else {
		/*if a, ok := a; ok {
			return a
		}*/
	}
	return a
}