//
package nn

import "io"

type jsonType string

//
func JSON(filename ...string) io.ReadWriter {
	/*if len(filename) > 0 && filename[0] != "" {
		return jsonType(filename[0])
	} else {
		return jsonType("")
	}*/
	return jsonType(filename[0])
}

func (j jsonType) Read(p []byte) (n int, err error) {
	return
}

func (j jsonType) Write(p []byte) (n int, err error) {
	return
}