package dataset

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/teratron/gonn/pkg/utils"
)

// Recognised IDX dtype bytes per the MNIST specification.
// (https://yann.lecun.com/exdb/mnist/ — IDX File Format).
const (
	idxDTypeUInt8  byte = 0x08
	idxDTypeInt8   byte = 0x09
	idxDTypeInt16  byte = 0x0B
	idxDTypeInt32  byte = 0x0C
	idxDTypeFloat  byte = 0x0D
	idxDTypeDouble byte = 0x0E
)

// maxIDXRecordLen caps the per-record element count parsed from an IDX header
// (256 Mi elements) so a hostile header cannot trigger a huge allocation or an
// integer overflow.
const maxIDXRecordLen = 256 << 20

// idxHeader is the parsed form of the leading bytes of an IDX file.
// dtype identifies the element type (0x08 = uint8 for MNIST images and
// labels), ndim is the number of dimensions, dims holds the per-dimension
// counts, count is dims[0] (number of records), and recLen is
// product(dims[1:]) — the flat element count per record.
//
// AI-Meta:
//   - Purpose: Parsed IDX file header carrying dtype, dimensions, record count, and per-record element length.
//   - Concurrency: Safe; value type, no mutation after readIDXHeader returns.
//   - Related: [readIDXHeader], [IDXReader], [MNISTLoader].
//   - Stability: Stable.
type idxHeader struct {
	dims   []int32
	count  int
	recLen int
	dtype  byte
	ndim   byte
}

// readIDXHeader reads the 4-byte magic + ndim*4 dimension bytes from r and
// returns the parsed header. The magic must be 0x00 0x00 dtype ndim with
// dtype in the recognised set; mismatches return [utils.ErrIDXMagic].
//
// AI-Meta:
//   - Purpose: Parse and validate the IDX file header per FMT-1 / FMT-2.
//   - Usage: hdr, err := readIDXHeader(r).
//   - Errors: ErrIDXMagic (bad magic / unknown dtype), ErrIO (short read).
//   - Concurrency: Safe; consumes r in a single read sequence.
//   - Related: [idxHeader], [IDXReader], [utils.ErrIDXMagic].
//   - Stability: Stable.
func readIDXHeader(r io.Reader) (idxHeader, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return idxHeader{}, utils.Wrap(utils.ErrIO, err, "IDX: read magic")
	}
	if magic[0] != 0x00 || magic[1] != 0x00 {
		return idxHeader{}, fmt.Errorf("IDX: invalid magic prefix [%#x %#x]: %w",
			magic[0], magic[1], utils.ErrIDXMagic)
	}
	dtype := magic[2]
	switch dtype {
	case idxDTypeUInt8, idxDTypeInt8, idxDTypeInt16, idxDTypeInt32,
		idxDTypeFloat, idxDTypeDouble:
		// ok
	default:
		return idxHeader{}, fmt.Errorf("IDX: unknown dtype %#x: %w",
			dtype, utils.ErrIDXMagic)
	}
	ndim := magic[3]
	if ndim == 0 {
		return idxHeader{}, fmt.Errorf("IDX: ndim must be ≥1, got 0: %w", utils.ErrIDXMagic)
	}
	dims := make([]int32, ndim)
	dimBuf := make([]byte, 4*int(ndim))
	if _, err := io.ReadFull(r, dimBuf); err != nil {
		return idxHeader{}, utils.Wrap(utils.ErrIO, err, "IDX: read dimensions")
	}
	for i := range dims {
		dims[i] = int32(binary.BigEndian.Uint32(dimBuf[i*4:]))
		if dims[i] < 0 {
			return idxHeader{}, fmt.Errorf("IDX: negative dimension %d at index %d: %w",
				dims[i], i, utils.ErrIDXMagic)
		}
	}
	// Compute the per-record element count with an overflow/size guard so a
	// crafted header (e.g. from `-download`ed data) cannot make NewIDXReader
	// allocate a gigantic per-record buffer or overflow int (audit E).
	recLen := 1
	for i := 1; i < int(ndim); i++ {
		recLen *= int(dims[i])
		if recLen < 0 || recLen > maxIDXRecordLen {
			return idxHeader{}, fmt.Errorf(
				"IDX: record length overflows sane bound (dim product > %d): %w",
				maxIDXRecordLen, utils.ErrIDXMagic)
		}
	}
	return idxHeader{
		dtype:  dtype,
		ndim:   ndim,
		dims:   dims,
		count:  int(dims[0]),
		recLen: recLen,
	}, nil
}

// elementSize returns the on-disk byte size of a single IDX element of dtype.
// Used to size the per-record read buffer.
func (h idxHeader) elementSize() int {
	switch h.dtype {
	case idxDTypeUInt8, idxDTypeInt8:
		return 1
	case idxDTypeInt16:
		return 2
	case idxDTypeInt32, idxDTypeFloat:
		return 4
	case idxDTypeDouble:
		return 8
	default:
		return 1
	}
}

// IDXReader streams records out of an IDX file one at a time. The buffer
// returned by Next is reused — callers MUST consume or copy the bytes before
// the next call (FMT-3 row-major streaming, bounded memory).
//
// AI-Meta:
//   - Purpose: Stream successive records from an IDX file; per-record bounded memory.
//   - Concurrency: NotSafe; Next mutates the shared read buffer.
//   - Related: [readIDXHeader], [MNISTLoader].
//   - Stability: Stable.
type IDXReader struct {
	r      io.Reader
	buf    []byte
	hdr    idxHeader
	cursor int // number of records already consumed
}

// NewIDXReader parses the header from r and returns a reader ready to stream
// records.
func NewIDXReader(r io.Reader) (*IDXReader, error) {
	hdr, err := readIDXHeader(r)
	if err != nil {
		return nil, err
	}
	return &IDXReader{
		r:   r,
		hdr: hdr,
		buf: make([]byte, hdr.recLen*hdr.elementSize()),
	}, nil
}

// Header exposes the parsed header to the caller.
func (ir *IDXReader) Header() idxHeader { return ir.hdr }

// Count returns the total record count from the header.
func (ir *IDXReader) Count() int { return ir.hdr.count }

// RecordLen returns the flat element count per record.
func (ir *IDXReader) RecordLen() int { return ir.hdr.recLen }

// Next reads the next record into the internal buffer and returns the buffer
// slice. Returns io.EOF after the last record. The returned slice aliases
// the reader's buffer; copy if you need to retain the data across calls.
func (ir *IDXReader) Next() ([]byte, error) {
	if ir.cursor >= ir.hdr.count {
		return nil, io.EOF
	}
	if _, err := io.ReadFull(ir.r, ir.buf); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, utils.Wrap(utils.ErrIO, err,
				"IDX: short read at record %d/%d", ir.cursor, ir.hdr.count)
		}
		return nil, utils.Wrap(utils.ErrIO, err, "IDX: read record")
	}
	ir.cursor++
	return ir.buf, nil
}

// Reset would require a seekable underlying reader; this implementation
// returns ErrUnsupported because we accept an io.Reader. Callers that need
// rewind should construct a new IDXReader from a freshly opened file.
func (ir *IDXReader) Reset() error {
	return ErrUnsupported
}

// MNISTLoader streams (input, target) batches from a paired IDX image file
// and IDX label file. Pixels are normalised to [0, 1] by dividing by norm
// (255 for raw uint8 MNIST). Labels are emitted as one-element slices
// containing the raw class index cast to T; callers wanting one-hot vectors
// supply that transform separately.
//
// MNISTLoader implements [Dataset]. The dataset is single-pass per
// constructor invocation — call Reset to rewind, which reopens the
// underlying files when they were obtained from NewMNISTLoaderFiles.
//
// MNISTLoader also implements [ImageShaper] when WithImageShape is applied or
// when the IDX header carries ndim==3 (images file has explicit H×W dims).
//
// AI-Meta:
//   - Purpose: Stream MNIST IDX image + label pairs as Dataset batches with normalised pixels.
//   - Concurrency: NotSafe; Next mutates internal cursors.
//   - Related: [Dataset], [ImageShaper], [NewMNISTLoader], [NewMNISTLoaderFiles], [IDXReader].
//   - Stability: Stable.
type MNISTLoader[T utils.Float] struct {
	norm      T
	images    *IDXReader
	labels    *IDXReader
	openFn    func() (*IDXReader, *IDXReader, []io.Closer, error)
	imagePath string
	labelPath string
	openFiles []io.Closer
	batchSize int
	// imgShape holds the explicit (C, H, W) set by WithImageShape.
	// Zero value means "derive from IDX header or fall back to isqrt".
	imgShape [3]int
}

// WithImageShape sets the (channels, height, width) layout for the image
// records in the loader. When applied, compile() reads ImageShape() to
// populate Config.InputC/InputH/InputW automatically without requiring a
// separate WithInputShape call.
//
// Validation deferred to ImageShape(): channels*height*width must equal the
// IDX record length at the point where the shape is first consumed.
//
// AI-Meta:
//   - Purpose: Attach a 2-D image shape to MNISTLoader for automatic compile() integration.
//   - Usage: ldr.WithImageShape(1, 28, 28).
//   - Related: [ImageShaper], [MNISTLoader.ImageShape].
//   - Stability: Stable.
func (m *MNISTLoader[T]) WithImageShape(channels, height, width int) *MNISTLoader[T] {
	m.imgShape = [3]int{channels, height, width}
	return m
}

// ImageShape implements [ImageShaper]. Returns the shape set by WithImageShape
// when non-zero. Falls back to the IDX header dims when the images file has
// ndim==3 (standard MNIST 3-D file: N×H×W for single-channel). When ndim==1
// and the record length is a perfect square, infers (1, S, S). Returns (0,0,0)
// when neither condition holds.
//
// AI-Meta:
//   - Purpose: Expose (C,H,W) for compile() auto-wiring; satisfies [ImageShaper].
//   - Usage: c, h, w := loader.ImageShape().
//   - Related: [ImageShaper], [WithImageShape].
//   - Stability: Stable.
func (m *MNISTLoader[T]) ImageShape() (channels, height, width int) {
	if m.imgShape[0] != 0 {
		return m.imgShape[0], m.imgShape[1], m.imgShape[2]
	}
	if m.images == nil {
		return 0, 0, 0
	}
	hdr := m.images.Header()
	if hdr.ndim == 3 {
		// Standard MNIST 3-D layout: dims = [N, H, W]; single channel.
		return 1, int(hdr.dims[1]), int(hdr.dims[2])
	}
	// Flat 1-D record: try isqrt inference for single-channel square images.
	if hdr.ndim == 1 || (hdr.ndim == 2 && hdr.dims[1] > 0) {
		s := mnistISqrt(hdr.recLen)
		if s*s == hdr.recLen {
			return 1, s, s
		}
	}
	return 0, 0, 0
}

// mnistISqrt returns the integer square root of n (largest k with k*k <= n).
func mnistISqrt(n int) int {
	if n <= 0 {
		return 0
	}
	s := 1
	for s*s <= n {
		s++
	}
	return s - 1
}

// NewMNISTLoader wraps existing IDX readers. The images reader must point at
// a 3-D IDX file (count × rows × cols) and the labels reader must point at a
// 1-D IDX file with the same count. Reset is unsupported in this mode.
//
// AI-Meta:
//   - Purpose: Wrap pre-opened IDX readers into a streaming MNIST dataset adapter.
//   - Usage: ldr, err := dataset.NewMNISTLoader[float32](imgR, lblR, 32, 255.0).
//   - Errors: ErrMNISTRecordMismatch (image/label counts differ), ErrUserConfig (bad batch / norm).
//   - Concurrency: NotSafe; caller owns the underlying readers.
//   - Related: [MNISTLoader], [NewMNISTLoaderFiles].
//   - Stability: Stable.
func NewMNISTLoader[T utils.Float](images, labels *IDXReader, batchSize int, norm T) (*MNISTLoader[T], error) {
	if images == nil || labels == nil {
		return nil, utils.Newf(utils.ErrUserConfig,
			"NewMNISTLoader: images and labels must be non-nil")
	}
	if images.Count() != labels.Count() {
		return nil, fmt.Errorf(
			"NewMNISTLoader: image count %d != label count %d: %w",
			images.Count(), labels.Count(), utils.ErrMNISTRecordMismatch)
	}
	if batchSize <= 0 {
		return nil, utils.NewSizeError("NewMNISTLoader.batchSize", batchSize, "positive")
	}
	if norm <= 0 {
		return nil, utils.Newf(utils.ErrUserConfig,
			"NewMNISTLoader: norm must be positive, got %v", norm)
	}
	return &MNISTLoader[T]{
		images:    images,
		labels:    labels,
		norm:      norm,
		batchSize: batchSize,
	}, nil
}

// NewMNISTLoaderFiles opens the named image / label IDX files and returns a
// loader that supports Reset (each Reset reopens both files).
//
// AI-Meta:
//   - Purpose: Open MNIST IDX files by path and return a Reset-capable streaming loader.
//   - Usage: ldr, err := dataset.NewMNISTLoaderFiles[float32]("train-images.idx", "train-labels.idx", 32, 255.0).
//   - Errors: ErrIO (file open / short read), ErrIDXMagic (bad header), ErrMNISTRecordMismatch.
//   - Concurrency: NotSafe.
//   - Related: [MNISTLoader], [NewMNISTLoader].
//   - Stability: Stable.
func NewMNISTLoaderFiles[T utils.Float](imagePath, labelPath string, batchSize int, norm T) (*MNISTLoader[T], error) {
	open := func() (*IDXReader, *IDXReader, []io.Closer, error) {
		img, err := os.Open(imagePath)
		if err != nil {
			return nil, nil, nil, utils.Wrap(utils.ErrIO, err, "MNIST: open %q", imagePath)
		}
		lbl, err := os.Open(labelPath)
		if err != nil {
			_ = img.Close()
			return nil, nil, nil, utils.Wrap(utils.ErrIO, err, "MNIST: open %q", labelPath)
		}
		imgR, err := NewIDXReader(img)
		if err != nil {
			_ = img.Close()
			_ = lbl.Close()
			return nil, nil, nil, err
		}
		lblR, err := NewIDXReader(lbl)
		if err != nil {
			_ = img.Close()
			_ = lbl.Close()
			return nil, nil, nil, err
		}
		return imgR, lblR, []io.Closer{img, lbl}, nil
	}
	imgR, lblR, closers, err := open()
	if err != nil {
		return nil, err
	}
	ldr, err := NewMNISTLoader(imgR, lblR, batchSize, norm)
	if err != nil {
		for _, c := range closers {
			_ = c.Close()
		}
		return nil, err
	}
	ldr.openFn = open
	ldr.openFiles = closers
	ldr.imagePath = imagePath
	ldr.labelPath = labelPath
	return ldr, nil
}

// Next implements [Dataset.Next]. Returns one Batch of at most batchSize
// records; the final batch may be shorter. Returns io.EOF when the source
// is exhausted.
func (m *MNISTLoader[T]) Next(ctx context.Context) (Batch[T], error) {
	if err := ctx.Err(); err != nil {
		return Batch[T]{}, err
	}
	inputs := make([][]T, 0, m.batchSize)
	targets := make([][]T, 0, m.batchSize)
	for i := 0; i < m.batchSize; i++ {
		raw, err := m.images.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Batch[T]{}, err
		}
		// Copy because the underlying buffer is reused across calls.
		input := make([]T, len(raw))
		for j, b := range raw {
			input[j] = T(b) / m.norm
		}
		lblRaw, err := m.labels.Next()
		if err == io.EOF {
			return Batch[T]{}, fmt.Errorf(
				"MNIST: image stream had more records than labels: %w",
				utils.ErrMNISTRecordMismatch)
		}
		if err != nil {
			return Batch[T]{}, err
		}
		if len(lblRaw) == 0 {
			return Batch[T]{}, fmt.Errorf("MNIST: empty label record: %w", utils.ErrIDXMagic)
		}
		target := []T{T(lblRaw[0])}
		inputs = append(inputs, input)
		targets = append(targets, target)
	}
	if len(inputs) == 0 {
		return Batch[T]{}, io.EOF
	}
	return Batch[T]{Inputs: inputs, Targets: targets}, nil
}

// Reset reopens the underlying files when the loader was constructed via
// [NewMNISTLoaderFiles]; otherwise returns [ErrUnsupported]. Previously
// opened file handles are closed before the new ones are wired in.
func (m *MNISTLoader[T]) Reset(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.openFn == nil {
		return ErrUnsupported
	}
	for _, c := range m.openFiles {
		_ = c.Close()
	}
	m.openFiles = nil
	imgR, lblR, closers, err := m.openFn()
	if err != nil {
		return err
	}
	m.images = imgR
	m.labels = lblR
	m.openFiles = closers
	return nil
}

// Close releases any file handles owned by the loader. Safe to call on
// loaders constructed via [NewMNISTLoader] (no-op).
//
// AI-Meta:
//   - Purpose: Release file handles owned by the loader; safe on BYO-reader loaders.
//   - Usage: defer loader.Close().
//   - Concurrency: NotSafe; serialise with Next / Reset.
//   - Related: [NewMNISTLoaderFiles].
//   - Stability: Stable.
func (m *MNISTLoader[T]) Close() error {
	var first error
	for _, c := range m.openFiles {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	m.openFiles = nil
	return first
}

// Len returns (count, true) — the total number of records reported by the
// image IDX header.
func (m *MNISTLoader[T]) Len() (int, bool) {
	return m.images.Count(), true
}

// Compile-time interface assertion.
var _ Dataset[float32] = (*MNISTLoader[float32])(nil)
