package embedding

import (
	"math"

	"github.com/teratron/gonn/pkg/utils"
)

// buildSinusoidalTable precomputes the Vaswani et al. 2017 sinusoidal positional
// encoding table of shape [seqLen × dmodel] stored as a flat row-major slice.
//
//	PE(pos, 2i)   = sin(pos / 10000^(2i/dmodel))
//	PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))
//
// The table is fixed (not learned); its values are in [-1, 1].
func buildSinusoidalTable[T utils.Float](seqLen, dmodel int) []T {
	table := make([]T, seqLen*dmodel)
	for pos := range seqLen {
		base := pos * dmodel
		for i := 0; i < dmodel; i += 2 {
			angle := float64(pos) / math.Pow(10000.0, float64(i)/float64(dmodel))
			table[base+i] = T(math.Sin(angle))
			if i+1 < dmodel {
				table[base+i+1] = T(math.Cos(angle))
			}
		}
	}
	return table
}
