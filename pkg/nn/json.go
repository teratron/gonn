//
package nn

import "io"

type jsonType string

func JSON(filename string) io.ReadWriter {
	return jsonType(filename)
}

func (j jsonType) Read(p []byte) (n int, err error) {
	return
}

func (j jsonType) Write(p []byte) (n int, err error) {
	return
}