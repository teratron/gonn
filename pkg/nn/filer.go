package nn

import "os"

// Filer
type Filer interface {
	ReadWriter
	getValue(key string) interface{}
}

// File
func File(filename string) *os.File {
	file, err := os.Create(filename)
	if err != nil {
		LogError(err)
		os.Exit(1)
	}
	return file
}
