package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type jsonString string

// JSON
func JSON(filename ...string) ReadWriter {
	if len(filename) > 0 {
		return jsonString(filename[0])
	}
	return jsonString("")
}

// ToString
func (j jsonString) ToString() string {
	return string(j)
}

// Read
func (j jsonString) Read(reader Reader) {
	filename := string(j)
	if len(filename) == 0 {
		LogError(fmt.Errorf("json: file json is missing"))
	}

	b, err := ioutil.ReadFile(filename)
	if err != nil {
		LogError(err)
	}

	if n, ok := reader.(NeuralNetwork); ok {
		var data interface{}
		if err = json.Unmarshal(b, &data); err != nil {
			LogError(fmt.Errorf("read unmarshal %w", err))
		}
		//fmt.Println(data.(map[string]interface{})["weights"])

		if value, ok := data.(map[string]interface{})["name"]; ok {
			if value == n.name() {
				if err = json.Unmarshal(b, &n); err != nil {
					LogError(fmt.Errorf("read unmarshal %w", err))
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
					LogError(fmt.Errorf("read json: %w", ErrNotRecognized))
					return
				}
			}
		}
	}
}

// Write
func (j jsonString) Write(writer ...Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*perceptron); ok {
			filename := string(j)
			if len(filename) == 0 {
				if len(n.jsonName) > 0 {
					filename = n.jsonName
				} else {
					// TODO: generate path and filename
					filename = "neural_network.json"
				}
			}
			if n.isTrain {
				n.Copy(Weight())
			} else {
				LogError(fmt.Errorf("json write: %w", ErrNotTrained))
			}
			if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
				LogError(fmt.Errorf("write %w", err))
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				LogError(err)
			}
		}
	} else {
		LogError(fmt.Errorf("%w json write", ErrEmpty))
	}
}
