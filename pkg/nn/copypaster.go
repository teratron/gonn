package nn

import "github.com/zigenzoog/gonn/pkg"

// Copy
func (n *NN) Copy(copier pkg.Copier) {
	copier.Copy(n)
}

// Paste
func (n *NN) Paste(paster pkg.Paster) error {
	return paster.Paste(n)
}