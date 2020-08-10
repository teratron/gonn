package nn

import (
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/zigenzoog/gonn/pkg"
	"io"
	"io/ioutil"
	"log"
	"os"
)

type jsonType string

//
func JSON(filename ...string) pkg.ReaderWriter {
	/*if len(filename) > 0 && filename[0] != "" {
		return jsonType(filename[0])
	} else {
		return jsonType("")
	}*/
	return jsonType(filename[0])
}

/*func (j jsonType) Read([]byte) (n int, err error) {
	return
}

func (j jsonType) Write([]byte) (n int, err error) {
	return
}*/

func (j jsonType) Read(pkg.Reader) {}
func (j jsonType) Write(...pkg.Writer) {}

func (n *nn) readJSON(filename string) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load json: ", err)
	}
	fmt.Println(string(b))
	//fmt.Println(b)

	/*dec := json.NewDecoder(bytes.NewReader(b))
	fmt.Println(dec)*/

	/*err = dec.Decode(t0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(dec)*/

	/*for {
		t, err := dec.Token()
		if err == io.EOF {
			break
		}else if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%T: %v", t, t)
		if dec.More() {
			fmt.Printf(" (more)")
		}
		fmt.Printf("\n")
	}*/

	var v interface{}
	err = json.Unmarshal(b, &v)
	data := v.(map[string]interface{})
	//fmt.Println(data)

	aa := data["architecture"].(map[string]interface{})
	//fmt.Println(aa)
	dd := aa["Parameters"]
	//fmt.Println(dd)

	st := os.Stdout
	enc := json.NewEncoder(st)
	enc.SetIndent("", "\t")
	err = enc.Encode(&dd)

	pp := &perceptronParameter{}
	dec := json.NewDecoder(bufio.NewReader(st))
	if err := dec.Decode(pp); err == io.EOF {
		return
	}
	fmt.Println(pp.Weights)


	//for key, value := range aa["perceptron"].(map[string]interface{}) {
	/*for key, value := range aa {
		fmt.Printf("%s: %T - %v\n", key, value, value)
		for k, v := range value.(map[string]interface{}) {
			fmt.Printf("%s: %T - %v\n", k, v, v)
		}
	}*/

	/*err = json.Unmarshal(b, n)
	if err != nil {
		log.Fatal("Invalid format ", err)
	}*/
	//fmt.Printf("%T - %v", n, n.Architecture.Bias())
	//fmt.Println(n)
	//fmt.Println(n.Architecture)
	/*fmt.Println(n.Architecture.(*perceptron).Settings["perceptron"])
	if a, ok := n.Architecture.(*perceptron); ok {
		for key, value := range a.Settings {
			fmt.Printf("%s: %T - %v", key, value, value)
		}
	}*/

}

func (n *nn) writeJSON(filename string) {
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}