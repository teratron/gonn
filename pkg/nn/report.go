//
package nn

import (
	"io"
	"os"
)

type report struct {
	file	*os.File
	args	[]interface{}
}

func Report(file *os.File, args ...interface{}) io.Writer {
	return &report{file, args}
}

/*func (r *report) Read(p []byte) (n int, err error) {
	return
}*/

func (r *report) Write(p []byte) (n int, err error) {
	return
}
