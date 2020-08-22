//
package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

/*type ReaderWriter interface {
	Reader
	Writer
}

type Reader interface {
	pkg.Reader
	//readJSON(interface{})
	//readXML(string)
	//readCSV(string)
}

type Writer interface {
	pkg.Writer
	//writeJSON(string)
	//writeXML(string)
	//writeCSV(string)
}*/

// Read
func (n *NN) Read(reader pkg.Reader) {
	switch r := reader.(type) {
	case jsonType:
		n.readJSON(string(r))
	case xmlType:
		n.readXML(string(r))
	/*case csvType:
		n.readCSV(string(r))
	case dbType:
		n.readDB(r)*/
	default:
		if v, ok := r.(pkg.Reader); ok {
			v.Read(n)
		}
		//pkg.Log("This type is missing for read", true) // !!!
		//log.Printf("\tWrite: %T %v\n", r, r) // !!!
	}
}

// Write
func (n *NN) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		for _, w := range writer {
			switch v := w.(type) {
			case jsonType:
				n.writeJSON(string(v))
			case xmlType:
				n.writeXML(string(v))
			/*case csvType:
				n.writeCSV(string(v))
			case dbType:
				n.writeDB(string(v))*/
			default:
				if u, ok := v.(pkg.Writer); ok {
					u.Write(n)
				}
			}
		}
	} else {
		log.Println("Empty write") // !!!
	}
}