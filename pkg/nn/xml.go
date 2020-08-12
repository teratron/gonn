//
package nn

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type xmlType string

//
func XML(filename ...string) pkg.ReaderWriter {
	return xmlType(filename[0])
}

/*func (x xmlType) Read(p []byte) (n int, err error) {
	return
}

func (x xmlType) Write(p []byte) (n int, err error) {
	return
}*/

func (x xmlType) Read(pkg.Reader) {}
func (x xmlType) Write(...pkg.Writer) {}

func (n *NN) readXML(filename string) {
	/*t := test
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}
	err = json.Unmarshal(b, &t)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}*/
}

func (n *NN) writeXML(filename string) {
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("XML marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}