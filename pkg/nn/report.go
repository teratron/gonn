package nn

import (
	"os"

	"github.com/teratron/gonn/pkg"
)

// report
type report struct {
	file *os.File
	args []interface{}

	pkg.Writer
}

// Report
func Report(file *os.File, args ...interface{}) pkg.Writer {
	return &report{
		file: file,
		args: args,
	}
}
