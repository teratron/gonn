//
package nn

import "io"

type reportType struct {
	input	[]float64
	args	[]interface{}
}

func Report(input []float64, args ...interface{}) io.Writer {
	return &reportType{input, args}
}

func (r *reportType) Write(p []byte) (n int, err error) {
	return
}
