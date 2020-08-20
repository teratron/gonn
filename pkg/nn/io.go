//
package nn

import (
	"fmt"
	"log"
	"os"
)

//type fileType io.ReadWriter

//
func File(filename string) *os.File {
	//os.IsExist()
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("-----------------------Open----------------------------")
		file, err = os.Open(filename)
		if err != nil {
			log.Fatal("Error !!!", err)
		}
	}
	return file
}