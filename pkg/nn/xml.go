//
package nn

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"os"
)

type xmlType string

//
func XML(filename ...string) io.ReadWriter {
	return xmlType(filename[0])
}

func (x xmlType) Read(p []byte) (n int, err error) {
	return
}

func (x xmlType) Write(p []byte) (n int, err error) {
	return
}

func (n *net) readXML(filename string) {
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

func (n *net) writeXML(filename string) {
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("XML marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}