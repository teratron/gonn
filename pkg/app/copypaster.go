package app

import "github.com/zigenzoog/gonn/pkg"

// Copy
func (a *app) Copy(copier pkg.Copier) {
	copier.Copy(a)
}

// Paste
func (a *app) Paste(paster pkg.Paster) {
	paster.Paste(a)
}
