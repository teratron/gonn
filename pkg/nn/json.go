//
package nn

import (
	"io"
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

/*func (n *NN) readJSON(filename string) {
	t := test
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load settings: ", err)
	}
	err = json.Unmarshal(b, &t)
	if err != nil {
		log.Fatal("Invalid settings format: ", err)
	}

	err = ioutil.WriteFile(filename, b, os.ModePerm)
}*/

/*func (n *NN) writeJSON(filename string) {
	//fmt.Println("+-+--++++-")
	//s := new(settings)
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}*/