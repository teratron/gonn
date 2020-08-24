package nn

import "github.com/zigenzoog/gonn/pkg"

// Copy
func (n *NN) Copy(copier pkg.Copier) {
	if c, ok := copier.(pkg.CopyPaster); ok {
		c.Copy(n)
	}
}

// Paste
func (n *NN) Paste(paster pkg.Paster) (err error) {
	if p, ok := paster.(pkg.CopyPaster); ok {
		err = p.Paste(n)
	}
	return
}