package main

import (
	"context"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"github.com/teratron/gonn/pkg/dataset"
	"github.com/teratron/gonn/pkg/nn"
	"github.com/teratron/gonn/pkg/utils"
)

// csvStreamThreshold is the file size beyond which the streaming CSV reader
// is used instead of a full in-memory load. Tests may override this value.
var csvStreamThreshold int64 = 64 * 1024 * 1024 // 64 MB

// loadSamples reads a CSV file into memory as a slice of nn.Sample[T].
// For files larger than csvStreamThreshold it drains the streaming dataset
// reader row-by-row to avoid holding the full raw string table in memory.
// Columns: first inputCols → Input, remaining → Target.
func loadSamples[T utils.Float](path string, inputCols, outputCols int) ([]nn.Sample[T], error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, utils.Wrap(utils.ErrIO, err, "loadSamples: stat %q", path)
	}

	if info.Size() > csvStreamThreshold {
		return loadSamplesStreaming[T](path, inputCols, outputCols)
	}
	return loadSamplesFull[T](path, inputCols, outputCols)
}

// loadSamplesFull reads all CSV rows at once via csv.ReadAll — preferred for
// files that comfortably fit in memory (≤ csvStreamThreshold).
func loadSamplesFull[T utils.Float](path string, inputCols, outputCols int) ([]nn.Sample[T], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, utils.Wrap(utils.ErrIO, err, "loadSamplesFull: open %q", path)
	}
	defer func() { _ = f.Close() }()

	r := csv.NewReader(f)
	r.FieldsPerRecord = inputCols + outputCols
	records, err := r.ReadAll()
	if err != nil {
		return nil, utils.Wrap(utils.ErrInputData, err, "loadSamplesFull: read %q", path)
	}

	return recordsToSamples[T](records, inputCols, outputCols, path)
}

// loadSamplesStreaming uses pkg/dataset.NewCSVDataset to parse the file
// lazily and drains every batch into the result slice.
func loadSamplesStreaming[T utils.Float](path string, inputCols, outputCols int) ([]nn.Sample[T], error) {
	ds, err := dataset.NewCSVDataset[T](path, inputCols, outputCols, 256)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	var samples []nn.Sample[T]
	for {
		batch, err := ds.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		for i := range batch.Inputs {
			samples = append(samples, nn.Sample[T]{Input: batch.Inputs[i], Target: batch.Targets[i]})
		}
	}
	return samples, nil
}

// recordsToSamples converts raw csv.ReadAll output to []nn.Sample[T].
// Returns ErrInputData (with row index) on any parse failure.
func recordsToSamples[T utils.Float](records [][]string, inputCols, outputCols int, path string) ([]nn.Sample[T], error) {
	total := inputCols + outputCols
	samples := make([]nn.Sample[T], 0, len(records))

	for row, rec := range records {
		if len(rec) != total {
			return nil, utils.Newf(utils.ErrInputData,
				"%q row %d: expected %d columns, got %d", path, row+1, total, len(rec))
		}

		input := make([]T, inputCols)
		for i := range inputCols {
			v, err := strconv.ParseFloat(rec[i], 64)
			if err != nil {
				return nil, utils.Newf(utils.ErrInputData,
					"%q row %d col %d: %v", path, row+1, i+1, err)
			}
			input[i] = T(v)
		}

		target := make([]T, outputCols)
		for i := range outputCols {
			v, err := strconv.ParseFloat(rec[inputCols+i], 64)
			if err != nil {
				return nil, utils.Newf(utils.ErrInputData,
					"%q row %d col %d: %v", path, row+1, inputCols+i+1, err)
			}
			target[i] = T(v)
		}

		samples = append(samples, nn.Sample[T]{Input: input, Target: target})
	}
	return samples, nil
}
