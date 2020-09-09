//
package nn

import (
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/zigenzoog/gonn/pkg"
)

type csvString string

// CSV
func CSV(filename ...string/*, comma rune*/) pkg.ReadWriter {
	if len(filename) > 0 {
		return csvString(filename[0])
	} else {
		return csvString("")
	}
}

// Read
func (c csvString) Read(reader pkg.Reader) {
	/*if n, ok := reader.(*nn); ok {
		filename := string(c)
		if len(filename) == 0 {
			errCSV(fmt.Errorf("error: file csv is missing"))
		}
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			errOS(err)
		}
	}*/
}

// Write
func (c csvString) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*nn); ok {
			filename := string(c)
			if len(filename) == 0 {
				if len(n.csv) > 0 {
					filename = n.csv
				} else {
					// TODO: generate filename
					filename = "config/neural_network.csv"
				}
			}
			if n.IsTrain {
				n.Copy(Weight())
			} else {
				errNN(fmt.Errorf("csv write: %w", ErrNotTrained))
			}
			if b, err := csv.MarshalIndent(&n, "", "\t"); err != nil {
				errCSV(fmt.Errorf("write %w", err))
			} else if err = ioutil.WriteFile(filename, b, os.ModePerm); err != nil {
				errOS(err)
			}
		}
	} else {
		errNN(fmt.Errorf("%w csv write", ErrEmpty))
	}
}

// errCSV
func errCSV(err error) {
	switch e := err.(type) {
	case *csv.ParseError:
		log.Println("parse csv error:", e, "line:", e.Line, "column:", e.Column)
	default:
		log.Println(err)
	}
	//os.Exit(1)
}