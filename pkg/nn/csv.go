//
package nn

import "io"

type csvType string

//
func CSV(filename ...string) io.ReadWriter {
	return csvType(filename[0])
}

func (c csvType) Read(p []byte) (n int, err error) {
	return
}

func (c csvType) Write(p []byte) (n int, err error) {
	return
}