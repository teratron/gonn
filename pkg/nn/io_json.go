package nn

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type jsonType string

// JSON
func JSON(filename ...string) pkg.ReaderWriter {
	return jsonType(filename[0])
}

// Read
func (j jsonType) Read(reader pkg.Reader) {
	if r, ok := reader.(pkg.Reader); ok {
		r.Read(j)
	}
}

// Write
func (j jsonType) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if w, ok := writer[0].(pkg.Writer); ok {
			w.Write(j)
		}
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}

// readJSON
func (n *NN) readJSON(value interface{}) {
	filename, ok := value.(string)
	if ok {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			log.Fatal("Can't load json: ", err)
		}
		//fmt.Println(string(b))

		// Декодируем json в NN
		err = json.Unmarshal(b, &n)
		if err != nil {
			log.Println(err)
		}
		//fmt.Println(n)
		n.Architecture = nil
		n.IsInit       = false
		n.json         = filename

		// Декодируем json в тип map[string]interface{}
		var data interface{}
		err = json.Unmarshal(b, &data)
		if err != nil {
			log.Println(err)
		}
		//fmt.Println(data)

		for key, value := range data.(map[string]interface{}) {
			if v, ok := value.(map[string]interface{}); ok {
				if key == "architecture" {
					b, err = json.Marshal(&v)
					if err != nil {
						log.Println(err)
					}
					//fmt.Println(string(b))

					err = json.Unmarshal(b, &data)
					if err != nil {
						log.Println(err)
					}
					//fmt.Println(data)

					for k, v := range data.(map[string]interface{}) {
						//fmt.Printf("%s - %T - %v\n", k, v, v)
						switch k {
						case "perceptron":
							n.Architecture = &perceptron{
								Architecture: n,
							}
							if a, ok := n.Architecture.(*perceptron); ok {
								a.readJSON(v)
							}
						case "hopfield":
							n.Architecture = &hopfield{
								Architecture: n,
							}
							if a, ok := n.Architecture.(*hopfield); ok {
								a.readJSON(v)
							}
						default:
						}
					}
				}
			}
		}
	}
}

// writeJSON
func (n *NN) writeJSON(filename string) {
	if n.IsTrain {
		n.Copy(Weight())
	} else {
		log.Println("Not trained network")
	}
	if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
		log.Fatal("JSON marshaling failed: ", err)
	} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
		log.Fatal("Can't write file:", err)
	}
}