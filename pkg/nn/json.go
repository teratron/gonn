package nn

import (
	"encoding/json"
	"fmt"
	_ "io"
	"io/ioutil"
	"os"

	"github.com/teratron/gonn/pkg"
)

type jsonString string

// JSON
func JSON(filename ...string) pkg.ReadWriter {
	if len(filename) > 0 {
		return jsonString(filename[0])
	}
	return jsonString("")
}

// Read
func (j jsonString) Read(reader pkg.Reader) {
	filename := string(j)
	if len(filename) == 0 {
		pkg.LogError(fmt.Errorf("json: file json is missing"))
	}

	b, err := ioutil.ReadFile(filename)
	if err != nil {
		pkg.LogError(err)
	}

	if n, ok := reader.(NeuralNetwork); ok {
		var data interface{}
		if err = json.Unmarshal(b, &data); err != nil {
			pkg.LogError(fmt.Errorf("read unmarshal %w", err))
		}
		//fmt.Println(data.(map[string]interface{})["weights"])

		if value, ok := data.(map[string]interface{})["name"]; ok {
			if value == n.name() {
				if err = json.Unmarshal(b, &n); err != nil {
					pkg.LogError(fmt.Errorf("read unmarshal %w", err))
				}
				n.setStateInit(false)
				n.setNameJSON(filename)
			} else {
				switch value.(string) {
				case perceptronName:
					n = &perceptron{
						Name: perceptronName,
					}
					//fmt.Println(n.Architecture.(*perceptron).Weights)
				case hopfieldName:
					n = &hopfield{
						Name: hopfieldName,
					}
				default:
					pkg.LogError(fmt.Errorf("read json: %w", pkg.ErrNotRecognized))
					return
				}
			}
		}
	}
}

// Write
func (j jsonString) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*perceptron); ok {
			filename := string(j)
			if len(filename) == 0 {
				if len(n.nameJSON) > 0 {
					filename = n.nameJSON
				} else {
					// TODO: generate path and filename
					filename = "neural_network.json"
				}
			}
			if n.isTrain {
				n.Copy(Weight())
			} else {
				pkg.LogError(fmt.Errorf("json write: %w", pkg.ErrNotTrained))
			}
			if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
				pkg.LogError(fmt.Errorf("write %w", err))
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				pkg.LogError(err)
			}
		}
	} else {
		pkg.LogError(fmt.Errorf("%w json write", pkg.ErrEmpty))
	}
}
