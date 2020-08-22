package nn

import (
	"log"
	"os"
)

// File
func File(filename string) *os.File {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error !!!", err)
	}
	return file
}