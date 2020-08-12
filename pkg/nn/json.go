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

func (n *NN) readJSON(filename string) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatal("Can't load json: ", err)
	}
	//fmt.Println(string(b))

	n.json = filename

	// Декодируем json в NN
	err = json.Unmarshal(b, &n)
	if err != nil { log.Println(err) }
	//fmt.Println(n)

	// Декодируем json в map[string]interface{}
	var data interface{}
	err = json.Unmarshal(b, &data)
	if err != nil { log.Println(err) }
	//fmt.Println(data)

	for key, value := range data.(map[string]interface{}) {
		if v, ok := value.(map[string]interface{}); ok {
			if key == "architecture" {
				b, err = json.Marshal(&v)
				if err != nil { log.Println(err) }
				fmt.Println(string(b))

				err = json.Unmarshal(b, &data)
				if err != nil { log.Println(err) }
				//fmt.Println(data)

				for k, v := range data.(map[string]interface{}) {
					switch k {
					case "perceptron":
						b, err = json.Marshal(&v)
						if err != nil { log.Println(err) }
						fmt.Println(string(b))

						err = json.Unmarshal(b, &n.Architecture.(*perceptron).Configuration)
						if err != nil { log.Println(err) }
						fmt.Println(&n.Architecture.(*perceptron).Configuration)

					case "hopfield":
						fmt.Printf("%s - %T - %v\n", k, v, v)
					default:
					}
				}
			}
		}
	}
	/*dec := json.NewDecoder(bufio.NewReader(st1))
	fmt.Println("+-+---",st1)
	if _, ok := value.(map[string]interface{}); ok {
		err = dec.Decode(rr)
		err = json.Unmarshal(strings.(st1), &v)
		fmt.Printf("---- %T - %v\n", rr, rr)
	} else {
		err = dec.Decode(rr.IsTrain)
		fmt.Printf("---- %T - %v\n", rr.IsTrain, rr.IsTrain)
	}*/
	//fmt.Println(n.IsTrain)

	/*aa := data["architecture"].(map[string]interface{})
	//fmt.Println(aa)
	dd := aa["perceptron"]
	//fmt.Println(dd)

	st := os.Stdout
	enc := json.NewEncoder(st)
	enc.SetIndent("", "\t")
	err = enc.Encode(&dd)

	pp := &perceptron{}
	ppp := pp.Configuration
	dec := json.NewDecoder(bufio.NewReader(st))
	if err := dec.Decode(ppp); err == io.EOF {
		return
	}*/
	//fmt.Printf("%T - %v\n", ppp.Bias, ppp.Bias)

}

func (n *NN) writeJSON(filename string) {
	b, err := json.MarshalIndent(n, "", "\t")
	if err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	}
	err = ioutil.WriteFile(filename, b, os.ModePerm)
	if err != nil {
		log.Fatal("Can't write updated settings file:", err)
	}
}