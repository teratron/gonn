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

// JSON
func JSON(filename ...string) pkg.ReadWriter {
	if len(filename) > 0 {
		return jsonType(filename[0])
	} else {
		return jsonType("")
	}
}

// Read
func (j jsonType) Read(reader pkg.Reader) {
	if n, ok := reader.(*nn); ok {
		filename := string(j)
		if len(filename) == 0 {
			errJSON(fmt.Errorf("file json is missing\n"))
		}
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			errOS(err)
		}
		//fmt.Println(string(b))

		// Декодируем json в NN
		err = json.Unmarshal(b, &n)
		if err != nil {
			errJSON(err)
		}
		//fmt.Println(n)
		n.Architecture = nil
		n.IsInit = false
		n.json = filename

		// Декодируем json в тип map[string]interface{}
		var data interface{}
		err = json.Unmarshal(b, &data)
		if err != nil {
			errJSON(err)
		}
		//fmt.Println(data)

		for key, value := range data.(map[string]interface{}) {
			if v, ok := value.(map[string]interface{}); ok {
				if key == "architecture" {
					b, err = json.Marshal(&v)
					if err != nil {
						errJSON(err)
					}
					//fmt.Println(string(b))

					err = json.Unmarshal(b, &data)
					if err != nil {
						errJSON(err)
					}
					//fmt.Println(data)

					for k, v := range data.(map[string]interface{}) {
						switch k {
						case "perceptron":
							n.Architecture = &perceptron{Architecture: n}
							if a, ok := n.Architecture.(*perceptron); ok {
								a.readJSON(v)
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
		if n, ok := writer[0].(*nn); ok {
			filename := string(j)
			if len(filename) == 0 {
				if len(n.json) > 0 {
					filename = n.json
				} else {
					// TODO: generate filename
					filename = "config/neural_network.json"
				}
			}
			if n.IsTrain {
				n.Copy(Weight())
			} else {
				log.Println("network isn't trained")
				errNN(ErrNotTrained)
			}
			if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
				errJSON(err)
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				errOS(err)
			}
		}
	} else {
		pkg.Log("Empty write", true) // !!!
	}
}

// errJSON
func errJSON(err error) {
	switch e := err.(type) {
	case *json.SyntaxError:
		log.Println("syntax json error:", e, "offset:", e.Offset)
	case *json.UnmarshalTypeError:
		log.Println("unmarshal json error:", e, "offset:", e.Offset)
	case *json.MarshalerError:
		log.Println("marshaling json error:", e)
	default:
		log.Println("json error:", err)
	}
	os.Exit(1)
}
