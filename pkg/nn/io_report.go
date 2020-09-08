package nn

import (
	"fmt"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type report struct {
	file *os.File
	args []interface{}
}

// Report
func Report(file *os.File, args ...interface{}) pkg.Writer {
	return &report{file, args}
}

// Write
func (r *report) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*nn); ok {
			if a, ok := n.Architecture.(NeuralNetwork); ok {
				a.Write(r)
			}
		}
	} else {
		errNN(fmt.Errorf("%w write for report", ErrEmpty))
	}
}