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
		if os.IsNotExist(err) {
			log.Println("file not found:", filename)
			os.Exit(1)
		}
		log.Println("file open error:", err)
		os.Exit(1)
	}*/
	file, err := os.Create(filename)
	if err != nil {
		log.Println("file create error:", err)
		os.Exit(1)
	}
	return file
}