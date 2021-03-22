package util

import (
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v2"
)

type FileYAML struct {
	Name string
}

// Decode
func (y *FileYAML) Decode(data interface{}) error {
	file, err := os.OpenFile(y.Name, os.O_RDONLY, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = yaml.NewDecoder(file).Decode(data)
	}
	return err
}

// Encode
func (y *FileYAML) Encode(data interface{}) error {
	file, err := os.OpenFile(y.Name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = yaml.NewEncoder(file).Encode(data)
	}
	return err
}

// GetValue
func (y *FileYAML) GetValue(key string) interface{} {
	b, err := ioutil.ReadFile(y.Name)
	if err == nil {
		var data interface{}
		err = yaml.Unmarshal(b, &data)
		if err == nil {
			if value, ok := data.(map[interface{}]interface{})[key]; ok {
				return value
			}
			err = fmt.Errorf("key not found")
		}
	}
	return fmt.Errorf("yaml get value: %w", err)
}
