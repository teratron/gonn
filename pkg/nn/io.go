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
		errorOS(err)
	}
	return file
}

// errorOS
func errorOS(err error, args ...interface{}) {
	if len(args) > 0 {
		for _, a := range args {
			switch v := a.(type) {
			case string:
				if os.IsNotExist(err) {
					log.Println("file not found:", v)
				}
			default:
				log.Println("file error:", err, a)
			}
		}
	}
	switch e := err.(type) {
	case *os.LinkError:
		log.Println("link error:", e)
	case *os.PathError:
		log.Println("path error:", e)
	case *os.SyscallError:
		log.Println("syscall error:", e)
	default:
		if len(args) == 0 {
			log.Println("file error:", err)
		}
	}
	os.Exit(1)
}