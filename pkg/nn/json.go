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

func (j jsonString) toString() string {
	return string(j)
}

func (j jsonString) getValue(key string) interface{} {
	filename := string(j)
	if len(filename) == 0 {
		LogError(fmt.Errorf("json: file config is missing"))
		return nil
	}
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		LogError(err)
		return nil
	}
	var data interface{}
	if err = json.Unmarshal(b, &data); err != nil {
		LogError(fmt.Errorf("read unmarshal %w", err))
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
		LogError(fmt.Errorf("json: file config is missing"))
	}
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		LogError(fmt.Errorf("read json file %w", err))
	}
	if err = json.Unmarshal(b, &reader); err != nil {
		LogError(fmt.Errorf("json unmarshal %w", err))
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

			if b, err := json.MarshalIndent(&n, "", "\t"); err != nil {
				LogError(fmt.Errorf("json marshal %w", err))
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				LogError(fmt.Errorf("write json file %w", err))
			}
		}
	} else {
		LogError(fmt.Errorf("%w json write", ErrEmpty))
	}
}
