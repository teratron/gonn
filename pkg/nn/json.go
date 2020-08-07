//
package nn

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
)

type jsonType string

//
func JSON(filename ...string) io.ReadWriter {
	/*if len(filename) > 0 && filename[0] != "" {
		return jsonType(filename[0])
	} else {
		return jsonType("")
	}*/
	return jsonType(filename[0])
}

func (j jsonType) Read(p []byte) (n int, err error) {
	return
}

func (j jsonType) Write(p []byte) (n int, err error) {
	return
}

func (n *net) readJSON(filename string) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}
	err = json.Unmarshal(b, n)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}
	fmt.Println(n.Architecture.(*perceptron).Settings)
}

/*func (n *net) writeJSON(filename string) {
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}*/