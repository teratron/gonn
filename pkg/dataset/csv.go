// Package dataset — CSV streaming adapter.
//
// Implements [l2-streaming-impl] §5.1 / §6.2: NewCSVDataset reads the
// canonical training-CSV format on demand using only the standard library
// (encoding/csv, bufio). The reader is a true stream: rows are decoded
// lazily and Reset reopens the file from the beginning.
package dataset

import (
	"bufio"
	"context"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"github.com/teratron/gonn/pkg/utils"
)

// NewCSVDataset opens a CSV file at path and returns a streaming Dataset[T].
// Each row contains inputSize numeric columns followed by outputSize numeric
// columns; the reader splits them lazily on each Next call.
//
// Open errors are returned immediately; per-row decode failures surface from
// Next() wrapped with utils.ErrInputData so callers can route via errors.Is.
//
// AI-Meta:
//   - Purpose: Construct a lazy CSV-backed Dataset for large files that do not fit in memory.
//   - Usage: ds, err := dataset.NewCSVDataset[float32]("data.csv", 4, 1, 32).
//   - Errors: ErrIO (file not found or unreadable), ErrInputData (bad size args or row decode).
//   - Related: [Dataset], [NewSliceDataset], [Prefetch].
func NewCSVDataset[T utils.Float](path string, inputSize, outputSize, batchSize int) (Dataset[T], error) {
	if inputSize <= 0 {
		return nil, utils.NewSizeError("NewCSVDataset.inputSize", inputSize, "positive")
	}
	if outputSize <= 0 {
		return nil, utils.NewSizeError("NewCSVDataset.outputSize", outputSize, "positive")
	}
	if batchSize <= 0 {
		return nil, utils.NewSizeError("NewCSVDataset.batchSize", batchSize, "positive")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, utils.Wrap(utils.ErrIO, err, "NewCSVDataset: open %q", path)
	}
	ds := &csvDataset[T]{
		path:       path,
		inputSize:  inputSize,
		outputSize: outputSize,
		batch:      batchSize,
	}
	ds.bind(f)
	return ds, nil
}

type csvDataset[T utils.Float] struct {
	file       *os.File
	reader     *csv.Reader
	path       string
	inputSize  int
	outputSize int
	batch      int
}

// bind attaches a freshly-opened file handle and configures the underlying
// csv.Reader. Used by NewCSVDataset and Reset to share setup logic.
func (c *csvDataset[T]) bind(f *os.File) {
	c.file = f
	c.reader = csv.NewReader(bufio.NewReader(f))
	c.reader.FieldsPerRecord = c.inputSize + c.outputSize
	c.reader.ReuseRecord = true
}

// Next reads up to c.batch rows and returns them as a Batch[T]. End-of-file
// is signalled with io.EOF only after all already-decoded rows have been
// returned to the caller; partial batches at the tail are returned
// before the io.EOF on the subsequent call. The file handle is closed
// transparently when EOF is reached so callers do not leak descriptors
// when they iterate to completion.
func (c *csvDataset[T]) Next(ctx context.Context) (Batch[T], error) {
	if err := ctx.Err(); err != nil {
		return Batch[T]{}, err
	}
	if c.reader == nil {
		return Batch[T]{}, io.EOF
	}

	inputs := make([][]T, 0, c.batch)
	targets := make([][]T, 0, c.batch)

	for len(inputs) < c.batch {
		row, err := c.reader.Read()
		if err == io.EOF {
			if len(inputs) == 0 {
				c.closeFile()
				return Batch[T]{}, io.EOF
			}
			break
		}
		if err != nil {
			c.closeFile()
			return Batch[T]{}, utils.Wrap(utils.ErrInputData, err,
				"csvDataset: read row %d", len(inputs))
		}
		input, target, parseErr := parseRow[T](row, c.inputSize, c.outputSize)
		if parseErr != nil {
			c.closeFile()
			return Batch[T]{}, parseErr
		}
		inputs = append(inputs, input)
		targets = append(targets, target)
	}

	return Batch[T]{Inputs: inputs, Targets: targets}, nil
}

// closeFile releases the file descriptor held by the dataset and clears
// the reader. Subsequent Next calls return io.EOF until Reset is called.
func (c *csvDataset[T]) closeFile() {
	if c.file != nil {
		_ = c.file.Close()
		c.file = nil
	}
	c.reader = nil
}

// Reset closes the current file handle and reopens path so iteration
// restarts from the first row. Honoured ctx cancellation is checked
// before any IO.
func (c *csvDataset[T]) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.file != nil {
		_ = c.file.Close()
	}
	f, err := os.Open(c.path)
	if err != nil {
		return utils.Wrap(utils.ErrIO, err, "csvDataset: reopen %q", c.path)
	}
	c.bind(f)
	return nil
}

// Len returns (-1, false) because CSV streams cannot announce their length
// without scanning the whole file (DAT-1).
func (c *csvDataset[T]) Len() (int, bool) { return -1, false }

// parseRow splits one CSV record into the input vector and target vector
// using the configured widths and parses each cell with strconv.ParseFloat.
// All errors are wrapped with utils.ErrInputData so callers route via
// errors.Is per DAT-4.
func parseRow[T utils.Float](row []string, inSize, outSize int) ([]T, []T, error) {
	if len(row) != inSize+outSize {
		return nil, nil, utils.Newf(utils.ErrInputData,
			"csvDataset: row width %d != expected %d", len(row), inSize+outSize)
	}
	input := make([]T, inSize)
	target := make([]T, outSize)
	bitSize := floatBitSize[T]()
	for i := range inSize {
		v, err := strconv.ParseFloat(row[i], bitSize)
		if err != nil {
			return nil, nil, utils.Wrap(utils.ErrInputData, err,
				"csvDataset: parse input col %d (%q)", i, row[i])
		}
		input[i] = T(v)
	}
	for i := range outSize {
		v, err := strconv.ParseFloat(row[inSize+i], bitSize)
		if err != nil {
			return nil, nil, utils.Wrap(utils.ErrInputData, err,
				"csvDataset: parse target col %d (%q)", i, row[inSize+i])
		}
		target[i] = T(v)
	}
	return input, target, nil
}

// floatBitSize returns 32 for float32 generic specialisations and 64 for
// float64, controlling strconv.ParseFloat precision per cell.
func floatBitSize[T utils.Float]() int {
	var z T
	switch any(z).(type) {
	case float32:
		return 32
	default:
		return 64
	}
}
