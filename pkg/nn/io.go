//
package nn

import (
	"fmt"
	"io"
	"log"
	"os"
)

type fileType io.ReadWriter

func File(filename string) *os.File {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("open")
		file, err = os.Open(filename)
	}
	if err != nil {
		log.Fatal("Error !!!")
	}
	return file
}