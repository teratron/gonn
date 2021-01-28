package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type jsonString string

var defaultNameJSON = "./neural_network.json"

// JSON
func JSON(filename ...string) ReadWriter {
	if len(filename) > 0 {
		return jsonString(filename[0])
	}
	return jsonString("")
}

func (j jsonString) toString() string {
	return string(j)
}

func (j jsonString) getValue(key string) interface{} {
	filename := string(j)
	if len(filename) == 0 {
		LogError(fmt.Errorf("json: file is missing"))
		return nil
	}
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		LogError(fmt.Errorf("read json file %w", err))
		return nil
	}
	var data interface{}
	if err = json.Unmarshal(b, &data); err != nil {
		LogError(fmt.Errorf("json unmarshal %w", err))
		return nil
	}
	if value, ok := data.(map[string]interface{})[key]; ok {
		return value
	}
	return nil
}

// Read
func (j jsonString) Read(reader Reader) {
	filename := string(j)
	if len(filename) == 0 {
		LogError(fmt.Errorf("json: file is missing"))
		return
	}
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		LogError(fmt.Errorf("read json file %w", err))
		return
	}
	if err = json.Unmarshal(b, &reader); err != nil {
		LogError(fmt.Errorf("json unmarshal %w", err))
	}
}

// Write
func (j jsonString) Write(writer ...Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(NeuralNetwork); ok {
			b, err := json.MarshalIndent(&n, "", "\t")
			if err != nil {
				LogError(fmt.Errorf("json marshal %w", err))
			}
			filename := string(j)
			if len(filename) == 0 {
				if name := n.nameJSON(); len(name) > 0 {
					filename = name
				} else {
					filename = defaultNameJSON
				}
			}
			if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				LogError(fmt.Errorf("write json file %w", err))
			}
		}
	} else {
		LogError(fmt.Errorf("%w json write", ErrEmpty))
	}
}
