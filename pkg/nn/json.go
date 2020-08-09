package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
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

	//t1 := test1{Name: "1"}
	//t2 := test2{Name2: "2"}
	//t0 := new(test0)
	//t0 := &test0{}
	//fmt.Println(*t0)
	//_ = json.Unmarshal(b, &t0)
	//fmt.Printf("%T %v",*t0, t0)

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

	err = json.Unmarshal(b, n)
	if err != nil {
		log.Fatal("Invalid format ", err)
	}
	fmt.Printf("%T - %v", n, n.Architecture)
	/*fmt.Println(n.Architecture.(*perceptron).Settings["perceptron"])
	if a, ok := n.Architecture.(*perceptron); ok {
		for key, value := range a.Settings {
			fmt.Printf("%s: %T - %v", key, value, value)
		}
	}*/

}

func (n *nn) writeJSON(filename string) {

	//v := Parameters(n.Get().(*perceptron))

	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}