//
package nn

import "io"

type dbType string

func DB(db string, filename ...string) io.Reader {
	return nil
}

func (d dbType) Read(p []byte) (n int, err error) {
	return
}

func (d dbType) Write(p []byte) (n int, err error) {
	return
}