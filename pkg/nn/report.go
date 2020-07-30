//
package nn

import (
	"fmt"
	"io"
	"os"
)

type report struct {
	writer	*os.File
	input	[]float64
	args	[]interface{}
}

func Report(writer *os.File, input []float64, args ...interface{}) io.Writer {
	fmt.Println("++++", writer)
	return &report{writer, input, args}
}

func (r *report) Write(p []byte) (n int, err error) { return }
