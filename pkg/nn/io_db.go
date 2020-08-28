//
package nn

import (
	"github.com/zigenzoog/gonn/pkg"
)

type dbType string

func DB(db string, filename ...string) pkg.Reader {
	return nil
}

/*func (d dbType) Read(p []byte) (n int, err error) {
	return
}

func (d dbType) Write(p []byte) (n int, err error) {
	return
}*/

func (d dbType) Read(pkg.Reader)     {}
func (d dbType) Write(...pkg.Writer) {}
