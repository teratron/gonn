package nn

import (
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

/*var (
	fileIn  = flag.String("in",  "","Specify input file path.")
	fileOut = flag.String("out", "","Specify output file path.")
)*/

// Filer
type Filer interface {
	pkg.ReadWriter
}

// File
func File(filename string) *os.File {
	/*file, err := os.Open(filename)
	if err != nil {
		errorOS(err, filename)
	}*/
	file, err := os.Create(filename)
	if err != nil {
		errOS(err)
	}
	return file
}

// errOS
func errOS(err error) {
	switch e := err.(type) {
	case *os.LinkError:
		log.Println("link error:", e)
	case *os.PathError:
		log.Println("path error:", e)
	case *os.SyscallError:
		log.Println("syscall error:", e)
	default:
		log.Println("os error:", err)
	}
	os.Exit(1)
}
