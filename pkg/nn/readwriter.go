package nn

import (
	"log"

	"github.com/zigenzoog/gonn/pkg"
)

// Read
func (n *nn) Read(reader pkg.Reader) {
	reader.Read(n)
}

// Write
func (n *nn) Write(writer ...pkg.Writer) {
	if len(writer) > 0 {
		for _, w := range writer {
			w.Write(n)
		}
	} else {
		log.Println("Empty write") // !!!
	}
}
