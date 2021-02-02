package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
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

func (j jsonString) fileName() (filename string, err error) {
	filename = string(j)
	if len(filename) == 0 {
		err = ErrNoFile
	}
	return
}

func (j jsonString) getValue(key string) interface{} {
	var (
		filename string
		err      error
		b        []byte
		data     interface{}
	)
	filename, err = j.fileName()
	if err == nil {
		b, err = ioutil.ReadFile(filename)
		if err == nil {
			err = json.Unmarshal(b, &data)
			if err == nil {
				if value, ok := data.(map[string]interface{})[key]; ok {
					return value
				}
				err = fmt.Errorf("key not found")
			}
		}
	}
	err = fmt.Errorf("json get value: %w", err)
	log.Println(err)
	return err
}

// Read
func (j jsonString) Read(reader Reader) (err error) {
	var (
		filename string
		b        []byte
	)
	filename, err = j.fileName()
	if err == nil {
		b, err = ioutil.ReadFile(filename)
		if err == nil {
			err = json.Unmarshal(b, &reader)
		}
	}
	if err != nil {
		err = fmt.Errorf("json read: %w", err)
		log.Println(err)
	}
	return
}

var defaultNameJSON = "./neural_network.json"

// Write
func (j jsonString) Write(writer ...Writer) (err error) {
	var b []byte
	if len(writer) > 0 {
		if n, ok := writer[0].(NeuralNetwork); ok {
			b, err = json.MarshalIndent(&n, "", "\t")
			if err == nil {
				filename := string(j)
				if len(filename) == 0 {
					if name := n.nameJSON(); len(name) > 0 {
						filename = name
					} else {
						filename = defaultNameJSON
					}
				}
				err = ioutil.WriteFile(filename, b, os.ModePerm)
			}
		}
	} else {
		err = ErrEmpty
	}
	if err != nil {
		err = fmt.Errorf("json write: %w", err)
		log.Println(err)
	}
	return
}
