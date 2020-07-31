//
package nn

import (
	"io"
	"os"
)

type report struct {
	file	*os.File
	size	*int64
	args	[]interface{}
}

func Report(file *os.File, args ...interface{}) io.Writer {
	info, _ := file.Stat()
	i := info.Size()
	return &report{file, &i, args}
}

func (r *report) Write(p []byte) (n int, err error) { return }
