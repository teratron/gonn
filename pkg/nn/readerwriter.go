//
package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

type readerWriter interface {
	reader
	writer
}

type reader interface {
	pkg.Reader
	readJSON(interface{})
	//readXML(string)
	//readCSV(string)
}

type writer interface {
	pkg.Writer
	writeJSON(string)
	//writeXML(string)
	//writeCSV(string)
}

// Read
func (n *NN) Read(reader pkg.Reader) {
	/*if a, ok := n.Get().(NeuralNetwork); ok {
		a.Read(reader)
	}*/
	if r, ok := reader.(pkg.Reader); ok {
		r.Read(n)
	}
	/*switch r := reader.(type) {
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
	}*/
}

// Write
func (n *NN) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		/*if a, ok := n.Get().(NeuralNetwork); ok {
			a.Write(writer...)
		}*/
		for _, w := range writer {
			if v, ok := w.(pkg.Writer); ok {
				v.Write(n)
			}
		}
		/*for _, w := range writer {
			switch v := w.(type) {
			case *report:
				if a, ok := n.Architecture.(NeuralNetwork); ok {
					a.Write(v)
				}
			case jsonType:
				n.writeJSON(string(v))
			case xmlType:
				n.writeXML(string(v))
			case csvType:
			case dbType:
			default:
				if a, ok := n.Architecture.(NeuralNetwork); ok {
					a.Write(writer...)
				}
			}
		}*/
	} else {
		log.Println("Empty write") // !!!
	}
}