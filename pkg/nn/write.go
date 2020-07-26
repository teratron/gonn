package nn

import (
	"io"
	"os"
)

func (n *nn) Write(writer ...io.Writer) {
	lenWriter := len(writer)
	if lenWriter > 0 {
		i := 0
		for _, w := range writer {
			if _, ok := w.(*os.File); ok { i++ }
		}
		if i < lenWriter {
			if a, ok := n.Get().(NeuralNetwork); ok {
				a.Write(writer...)
			}
		} else {
			Log("!!!", true)
		}
	} else {
	}
}