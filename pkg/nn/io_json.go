package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type jsonString string

// JSON
func JSON(filename ...string) pkg.ReadWriter {
	if len(filename) > 0 {
		return jsonString(filename[0])
	} else {
		return jsonString("")
	}
}

// Read
func (j jsonString) Read(reader pkg.Reader) {
	if n, ok := reader.(*nn); ok {
		filename := string(j)
		if len(filename) == 0 {
			errJSON(fmt.Errorf("json: file json is missing\n"))
		}
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			errOS(err)
		}
		//fmt.Println(string(b))

		// Декодируем json в NN
		err = json.Unmarshal(b, &n)
		if err != nil {
			errJSON(fmt.Errorf("--read unmarshal %w", err))
		}
		//fmt.Println(n)
		n.Architecture = nil
		n.IsInit = false
		n.json = filename

		// Декодируем json в тип map[string]interface{}
		var data interface{}
		err = json.Unmarshal(b, &data)
		if err != nil {
			errJSON(fmt.Errorf("read unmarshal %w", err))
		}
		//fmt.Println(data)

		for key, value := range data.(map[string]interface{}) {
			if v, ok := value.(map[string]interface{}); ok {
				if key == "architecture" {
					b, err = json.Marshal(&v)
					if err != nil {
						errJSON(fmt.Errorf("read marshal %w", err))
					}
					//fmt.Println(string(b))

					err = json.Unmarshal(b, &data)
					if err != nil {
						errJSON(fmt.Errorf("read unmarshal %w", err))
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
							errNN(ErrNotRecognized)
						}
					}
				}
			}
		}
	}
}

// Write
func (j jsonString) Write(writer ...pkg.Writer) {
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
				errNN(ErrNotTrained)
			}
			if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
				errJSON(fmt.Errorf("write %w", err))
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				errOS(err)
			}
		}
	} else {
		errNN(ErrEmptyWrite)
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
		log.Println(err)
	}
	//os.Exit(1)
}
