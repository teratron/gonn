//
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

type csvType string

//
func CSV(filename ...string) pkg.ReadWriter {
	return csvType(filename[0])
}

/*func (c csvType) Read(p []byte) (n int, err error) {
	return
}

func (c csvType) Write(p []byte) (n int, err error) {
	return
}*/

func (c csvType) Read(pkg.Reader)     {}
func (c csvType) Write(...pkg.Writer) {}
