//
package nn

import "io"

type json struct {
	fileName string
}

func JSON(filename string) io.Reader {
	return nil
}

func (j json) Read(p []byte) (n int, err error) {
	return
}