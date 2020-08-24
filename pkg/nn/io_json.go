package nn

import (
	"encoding/json"
	"github.com/zigenzoog/gonn/pkg"
	"io/ioutil"
	"log"
	"os"
)

type jsonType string

// JSON
func JSON(filename ...string) pkg.ReadWriter {
	return jsonType(filename[0])
}

// Read
func (j jsonType) Read(reader pkg.Reader) {
	if n, ok := reader.(*NN); ok {
		filename := string(j)
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
						switch k {
						case "perceptron":
							n.Architecture = &perceptron{Architecture: n}
							if a, ok := n.Architecture.(*perceptron); ok {
								//fmt.Printf("%T %v", v, v)
								a.readJSON(v)
								//a.Read(pkg.Reader(v))
							}
						case "hopfield":
							n.Architecture = &hopfield{Architecture: n}
							if a, ok := n.Architecture.(*hopfield); ok {
								a.readJSON(v)
							}
						default:
							log.Println("Нейронная сеть не распознана")

						}
					}
				}
			}
		}
	}
}

// Write
func (j jsonType) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*NN); ok {
			filename := string(j)
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
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}