//
package nn

import (
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type report struct {
	file	*os.File
	args	[]interface{}
}

func Report(file *os.File, args ...interface{}) pkg.Writer {
	return &report{file, args}
}

func (r *report) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*NN); ok {
			if a, ok := n.Architecture.(NeuralNetwork); ok {
				a.Write(r)
			}
		}
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}