//
package nn

import (
	"fmt"
	"io"
	"os"
)

type fileType io.ReadWriter

func File(filename string) *os.File {
	file, err := os.Open(filename)
	if err == nil {
		file, err = os.Create(filename)
	}
	if err != nil {
		os.Exit(1)
	}
	//rwss := file
	rwss, _ := file.Stat()
	//fmt.Printf("`````````````` %v\n", rwss.Size())
	//info, _ := file.Stat()
	//i := info.Size()
	fmt.Printf("`````````````` %v\n", rwss)

	return file
}