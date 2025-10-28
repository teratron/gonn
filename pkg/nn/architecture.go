package nn

import (
	"fmt"
	"log"
	"strings"

	"github.com/teratron/gonn/pkg"
	"github.com/teratron/gonn/pkg/utils"
)

// Get.
func Get(reader string) NN[float32] {
	var err error
	r := utils.ReadFile(reader)
	if _, ok := r.(*utils.FileError); ok {
		switch strings.ToLower(reader) {
		case Perceptron:
			return perceptron[float32]()
		default:
			err = fmt.Errorf("neural network is %w", pkg.ErrNotRecognized)
		}
	} else {
		switch v := r.GetValue("name").(type) {
		case error:
			err = v
		case string:
			if n := Get(v); n != nil {
				if err = r.Decode(n); err == nil {
					r.ClearData()
					n.Init(r)
					return n
				}
			}
		}
	}

	if err != nil {
		log.Printf("architecture.Get: %v", err)
	}
	return nil
}
