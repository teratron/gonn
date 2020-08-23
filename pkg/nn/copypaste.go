package nn

import (
	"fmt"
	"github.com/zigenzoog/gonn/pkg"
)

// Copy
func (n *NN) Copy(obj pkg.Getter) {
	if g, ok := obj.(pkg.CopyPaster); ok {
		fmt.Println("***", g)
		g.Copy(n)
	}
}

// Paste
func (n *NN) Paste(obj pkg.Getter) (err error) {
	if g, ok := obj.(pkg.CopyPaster); ok {
		g.Copy(n)
	} else {
		//err.Error()
	}
	return
}