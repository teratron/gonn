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
	//wait := make(chan int64)
	/*go func() {
		rws, _ := file.Stat()
		size := rws.Size()
		sizePtr := &size
		fmt.Println("+++", size, sizePtr)
		//wait <- true
		//fmt.Println("+++")
		for *sizePtr == size {
			//fmt.Println("===", *sizePtr)
			if size != rws.Size() {

				break
			}
			num, opened := <- wait     // получаем данные из потока
			if !opened {
				break    // если поток закрыт, выход из цикла
			}
			fmt.Println(num)
		}
		fmt.Println("===", rws.Size())
	}()*/
	//<- wait

	// Calling Pipe method
	//pipeReader, pipeWriter := io.Pipe()

	// Using Fprint in go function to write
	// data to the file
	/*go func() {
		fmt.Fprint(pipeWriter, "GeeksforGeeks\nis\na\nCS-Portal.\n")

		// Using Close method to close write
		pipeWriter.Close()
	}()*/

	// Creating a buffer
	//buffer := new(bytes.Buffer)

	// Calling ReadFrom method and writing
	// data into buffer
	//buffer.ReadFrom(pipeReader)

	// Prints the data in buffer
	//fmt.Println("++++++", buffer.String(), pipeReader)
	//fmt.Println(file.Fd())
	//var wait <-chan bool
	//go func() { <- wait }()
	//defer func() { err = file.Close() }()
	return file
}