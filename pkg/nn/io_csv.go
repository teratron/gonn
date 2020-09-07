//
package nn

import (
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

func (c csvString) Read(pkg.Reader)     {}
func (c csvString) Write(...pkg.Writer) {}
