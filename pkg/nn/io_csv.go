//
package nn

import (
	"encoding/csv"
	"fmt"
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

type csvString string

// CSV
func CSV(filename ...string) pkg.ReadWriter {
	if len(filename) > 0 {
		return csvString(filename[0])
	} else {
		return csvString("")
	}
}

// Read
func (c csvString) Read(reader pkg.Reader) {
}

// Write
func (c csvString) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		if n, ok := writer[0].(*nn); ok {
			filename := c
			if len(filename) == 0 {
				if len(n.csv) > 0 {
					filename = n.csv
				} else {
					// TODO: generate path and filename
					filename = "config/neural_network_weights.csv"
				}
			}
			if n.IsTrain {
				if a, ok := n.Architecture.(NeuralNetwork); ok {
					a.Write(filename)
				}
			} else {
				errNN(fmt.Errorf("csv write: %w", ErrNotTrained))
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
}