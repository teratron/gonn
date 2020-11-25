package nn

import (
	"os"

	"github.com/teratron/gonn/pkg"
)

// File
func File(filename string) *os.File {
	return pkg.File(filename)
}
