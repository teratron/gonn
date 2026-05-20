package embedding

import (
	"math/bits"

	"github.com/teratron/gonn/pkg/utils"
)

// sparseGrad accumulates per-row gradients for an embedding table.
// Only rows touched during a forward pass are tracked; the bitset
// allows iter to visit O(unique IDs) rows rather than O(Rows) (EMB-9).
//
// Layout: Buf[row*Dmodel : (row+1)*Dmodel] holds the accumulated gradient
// for row; Touched[row/64] bit (row%64) is set when add is called for that row.
type sparseGrad[T utils.Float] struct {
	Buf     []T      // [Rows × Dmodel] flat, zero-initialised
	Touched []uint64 // ceil(Rows/64) words; bit r%64 of word r/64 set iff row r was touched
	NumSet  int      // count of distinct rows touched (informational)
	Rows    int
	Dmodel  int
}

// newSparseGrad allocates a sparseGrad ready for use.
func newSparseGrad[T utils.Float](rows, dmodel int) sparseGrad[T] {
	words := (rows + 63) / 64
	return sparseGrad[T]{
		Buf:    make([]T, rows*dmodel),
		Touched: make([]uint64, words),
		Rows:   rows,
		Dmodel: dmodel,
	}
}

// add accumulates grad into row r of the gradient buffer and marks r as touched.
// grad must have length Dmodel. No bounds check on r beyond the slice access itself.
func (sg *sparseGrad[T]) add(r int, grad []T) {
	word, bit := r/64, uint(r%64)
	if sg.Touched[word]&(1<<bit) == 0 {
		sg.Touched[word] |= 1 << bit
		sg.NumSet++
	}
	base := r * sg.Dmodel
	for i, v := range grad {
		sg.Buf[base+i] += v
	}
}

// iter calls fn for every row that has been touched, passing the row index and
// its gradient slice (length Dmodel). Rows are visited in ascending order.
func (sg *sparseGrad[T]) iter(fn func(row int, grad []T)) {
	for wi, word := range sg.Touched {
		for word != 0 {
			tz := bits.TrailingZeros64(word)
			r := wi*64 + tz
			fn(r, sg.Buf[r*sg.Dmodel:(r+1)*sg.Dmodel])
			word &^= 1 << uint(tz)
		}
	}
}

// reset clears the bitset and zeroes only the touched rows of Buf.
// This is O(unique IDs × Dmodel) rather than O(Rows × Dmodel).
func (sg *sparseGrad[T]) reset() {
	for wi, word := range sg.Touched {
		if word == 0 {
			continue
		}
		for word != 0 {
			tz := bits.TrailingZeros64(word)
			r := wi*64 + tz
			base := r * sg.Dmodel
			clear(sg.Buf[base : base+sg.Dmodel])
			word &^= 1 << uint(tz)
		}
		sg.Touched[wi] = 0
	}
	sg.NumSet = 0
}
