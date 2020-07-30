package nn

import (
	"io"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

func (n *NN) Write(writer ...io.Writer) {
	lenWriter := len(writer)
	if lenWriter > 0 {
		i := 0
		for _, w := range writer {
			if _, ok := w.(*os.File); ok {
				i++
			}
		}
		if i < lenWriter {
			if a, ok := n.Get().(NeuralNetwork); ok {
				a.Write(writer...)
			}
		} else {
			pkg.Log("!!!", true)
		}
	} else {
		panic("Empty Write()") // !!!
	}
}