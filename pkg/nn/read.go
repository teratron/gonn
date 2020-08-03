package nn

import (
	"github.com/zigenzoog/gonn/pkg"
	"io"
	"log"
)

func (n *NN) Read(reader io.Reader) {
	if a, ok := n.Get().(NeuralNetwork); ok {
		a.Read(reader)
	}

	switch r := reader.(type) {
	case jsonType:
		n.readJSON(string(r))
	case xmlType:
		//n.readXML(string(r))
	case csvType:
		//n.readCSV(string(r))
	/*case db:
		p.writeDB(v)*/
	default:
		pkg.Log("This type is missing for read", true) // !!!
		log.Printf("\tWrite: %T %v\n", r, r) // !!!
	}
}

func (n *NN) readJSON(filename string) {
	/*t := test
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}
	err = json.Unmarshal(b, &t)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}

	err = ioutil.WriteFile(filename, b, os.ModePerm)*/

}