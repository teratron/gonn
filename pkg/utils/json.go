package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
)

type FileJSON struct {
	Name string
	Data []byte
	//Value map[string]interface{}
}

// Decode.
func (j *FileJSON) Decode(value interface{}) error {
	var r io.Reader
	var err error

	if len(j.Name) > 0 {
		r, err = os.OpenFile(j.Name, os.O_RDONLY, 0600)
		if err == nil {
			defer func() { err = r.(*os.File).Close() }()
		}
	} else if j.Data != nil {
		r = bytes.NewReader(j.Data)
	}
	err = json.NewDecoder(r).Decode(value)
	/*file, err := os.OpenFile(j.Name, os.O_RDONLY, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		err = json.NewDecoder(file).Decode(value)
	}*/
	return fmt.Errorf("utils.FileJSON.Decode: %w", err)
}

// Encode.
func (j *FileJSON) Encode(value interface{}) error {
	file, err := os.OpenFile(j.Name, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err == nil {
		defer func() { err = file.Close() }()
		e := json.NewEncoder(file)
		e.SetIndent("", "\t")
		err = e.Encode(value)
	}
	return fmt.Errorf("utils.FileJSON.Encode: %w", err)
}

// GetValue.
func (j *FileJSON) GetValue(key string) interface{} {
	var value interface{}
	var data []byte
	var err error

	if len(j.Name) > 0 {
		data, err = ioutil.ReadFile(j.Name)
		if err != nil {
			j.Name = ""
			goto ERROR
		}
	} else if j.Data != nil {
		data = j.Data
	}

	err = json.Unmarshal(data, &value)
	if err == nil {
		if k, ok := value.(map[string]interface{})[key]; ok {
			return k
		}
		err = fmt.Errorf("key not found")
	}

	/*data, err := ioutil.ReadFile(j.Name)
	if err == nil {
		var value interface{}
		err = json.Unmarshal(data, &value)
		if err == nil {
			if k, ok := value.(map[string]interface{})[key]; ok {
				return k
			}
			err = fmt.Errorf("key not found")
		}
	}*/
ERROR:
	return fmt.Errorf("utils.FileJSON.GetValue: %w", err)
}

// GetName.
func (j *FileJSON) GetName() string {
	return j.Name
}

// ClearData.
func (j *FileJSON) ClearData() {
	j.Data = nil
}
