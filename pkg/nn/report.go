package nn

import "os"

// report
type report struct {
	file *os.File
	args []interface{}

	Writer
}

// Report
func Report(file *os.File, args ...interface{}) Writer {
	return &report{
		file: file,
		args: args,
	}
}
