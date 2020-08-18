//
package nn

import (
	"fmt"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
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

//
func (n *NN) Read(reader pkg.Reader) {
	/*if a, ok := n.Get().(NeuralNetwork); ok {
		a.Read(reader)
	}*/
	switch r := reader.(type) {
	case jsonType:
		n.readJSON(string(r))
	case xmlType:
		//n.readXML(string(r))
	case csvType:
		//n.readCSV(string(r))
	case dbType:
		//n.readDB(r)
	default:
		pkg.Log("This type is missing for read", true) // !!!
		log.Printf("\tWrite: %T %v\n", r, r) // !!!
	}
}

//
func (n *NN) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		/*if a, ok := n.Get().(NeuralNetwork); ok {
			a.Write(writer...)
		}*/
		for _, w := range writer {
			switch v := w.(type) {
			case *report:
				if a, ok := n.Get().(NeuralNetwork); ok {
					a.Write(v)
				}
			case jsonType:
				n.writeJSON(string(v))
			case xmlType:
				n.writeXML(string(v))
			case csvType:
			case dbType:
			default:
				if a, ok := n.Get().(NeuralNetwork); ok {
					a.Write(writer...)
				}
			}
		}
	} else {
		log.Println("Empty Write()") // !!!
	}
}