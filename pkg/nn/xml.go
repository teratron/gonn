//
package nn

import "io"

type xmlType string

//
func XML(filename ...string) io.ReadWriter {
	return xmlType(filename[0])
}

func (x xmlType) Read(p []byte) (n int, err error) {
	return
}

func (x xmlType) Write(p []byte) (n int, err error) {
	return
}