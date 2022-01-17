package utils

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

type FileJSON struct {
	Name string
}

// Decode.
func (j *FileJSON) Decode(data interface{}) error {
	file, err := os.OpenFile(j.Name, os.O_RDONLY, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = json.NewDecoder(file).Decode(data)
	}
	return err
}

// Encode.
func (j *FileJSON) Encode(data interface{}) error {
	file, err := os.OpenFile(j.Name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		e := json.NewEncoder(file)
		e.SetIndent("", "\t")
		err = e.Encode(data)
	}
	return err
}

// GetValue.
func (j *FileJSON) GetValue(key string) interface{} {
	b, err := ioutil.ReadFile(j.Name)
	if err == nil {
		var data interface{}
		err = json.Unmarshal(b, &data)
		if err == nil {
			if value, ok := data.(map[string]interface{})[key]; ok {
				return value
			}
			err = fmt.Errorf("key not found")
		}
	}
	return fmt.Errorf("json get value: %w", err)
}

// GetName.
func (j *FileJSON) GetName() string {
	return j.Name
}
