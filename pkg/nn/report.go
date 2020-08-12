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

func (r *report) Write(...pkg.Writer) {}