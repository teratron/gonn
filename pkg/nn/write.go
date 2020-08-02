//
package nn

import (
	"io"
	"log"
)

//
func (n *NN) Write(writer ...io.Writer) {
	if len(writer) > 0 {
		if a, ok := n.Get().(NeuralNetwork); ok {
			a.Write(writer...)
		}
	} else {
		log.Println("Empty Write()") // !!!
	}
}