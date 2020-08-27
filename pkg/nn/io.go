package nn

import (
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

// Filer
type Filer interface {
	pkg.ReadWriter
}

// File
func File(filename string) *os.File {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Error !!!", err)
	}
	return file
}