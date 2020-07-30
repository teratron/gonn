//
package nn

import (
	"os"
)

type fileType *os.File

func File(filename string) *os.File {
	file, err := os.Create(filename)
	if err != nil { os.Exit(1) }
	//defer func() { err = file.Close() }()
	return file
}