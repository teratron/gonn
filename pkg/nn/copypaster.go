package nn

import "github.com/zigenzoog/gonn/pkg"

// Copy
func (n *nn) Copy(copier pkg.Copier) {
	copier.Copy(n)
}

// Paste
func (n *nn) Paste(paster pkg.Paster) error {
	return paster.Paste(n)
}
