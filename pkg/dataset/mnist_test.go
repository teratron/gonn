package dataset

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/teratron/gonn/pkg/utils"
)

// buildIDX builds an in-memory IDX byte stream with dtype=uint8 for testing.
// dims[0] is the record count; remaining dims describe one record's shape.
// data is the concatenated row-major bytes for all records.
func buildIDX(dims []int32, data []byte) []byte {
	if len(dims) == 0 || len(dims) > 255 {
		panic("buildIDX: bad ndim")
	}
	buf := &bytes.Buffer{}
	buf.WriteByte(0x00)
	buf.WriteByte(0x00)
	buf.WriteByte(idxDTypeUInt8)
	buf.WriteByte(byte(len(dims)))
	for _, d := range dims {
		_ = binary.Write(buf, binary.BigEndian, uint32(d))
	}
	buf.Write(data)
	return buf.Bytes()
}

func TestReadIDXHeaderValid(t *testing.T) {
	t.Parallel()
	raw := buildIDX([]int32{2, 3}, []byte{1, 2, 3, 4, 5, 6})
	hdr, err := readIDXHeader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("readIDXHeader err = %v", err)
	}
	if hdr.dtype != idxDTypeUInt8 {
		t.Errorf("dtype = %#x, want %#x", hdr.dtype, idxDTypeUInt8)
	}
	if hdr.count != 2 || hdr.recLen != 3 {
		t.Errorf("count/recLen = %d/%d, want 2/3", hdr.count, hdr.recLen)
	}
}

func TestReadIDXHeaderBadMagic(t *testing.T) {
	t.Parallel()
	bad := []byte{0x01, 0x00, idxDTypeUInt8, 0x01, 0, 0, 0, 1}
	_, err := readIDXHeader(bytes.NewReader(bad))
	if err == nil || !errors.Is(err, utils.ErrIDXMagic) {
		t.Errorf("err = %v, want ErrIDXMagic", err)
	}
}

func TestReadIDXHeaderUnknownDType(t *testing.T) {
	t.Parallel()
	bad := []byte{0x00, 0x00, 0x7F, 0x01, 0, 0, 0, 1}
	_, err := readIDXHeader(bytes.NewReader(bad))
	if err == nil || !errors.Is(err, utils.ErrIDXMagic) {
		t.Errorf("err = %v, want ErrIDXMagic", err)
	}
}

func TestReadIDXHeaderZeroNdim(t *testing.T) {
	t.Parallel()
	bad := []byte{0x00, 0x00, idxDTypeUInt8, 0x00}
	_, err := readIDXHeader(bytes.NewReader(bad))
	if err == nil || !errors.Is(err, utils.ErrIDXMagic) {
		t.Errorf("err = %v, want ErrIDXMagic", err)
	}
}

func TestReadIDXHeaderShortRead(t *testing.T) {
	t.Parallel()
	_, err := readIDXHeader(bytes.NewReader([]byte{0x00}))
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("err = %v, want ErrIO", err)
	}
}

func TestIDXReaderIterate(t *testing.T) {
	t.Parallel()
	raw := buildIDX([]int32{3, 2}, []byte{10, 20, 30, 40, 50, 60})
	rd, err := NewIDXReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("NewIDXReader: %v", err)
	}
	if rd.Count() != 3 || rd.RecordLen() != 2 {
		t.Errorf("Count/RecordLen = %d/%d, want 3/2", rd.Count(), rd.RecordLen())
	}
	want := [][]byte{{10, 20}, {30, 40}, {50, 60}}
	for i, expect := range want {
		got, err := rd.Next()
		if err != nil {
			t.Fatalf("Next #%d: %v", i, err)
		}
		if !bytes.Equal(got, expect) {
			t.Errorf("record %d = %v, want %v", i, got, expect)
		}
	}
	if _, err := rd.Next(); err != io.EOF {
		t.Errorf("after exhaustion err = %v, want io.EOF", err)
	}
	if err := rd.Reset(); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Reset err = %v, want ErrUnsupported", err)
	}
}

func TestIDXReaderShortRecord(t *testing.T) {
	t.Parallel()
	// Header promises 2 records of 4 bytes, but only 6 bytes of data follow.
	raw := buildIDX([]int32{2, 4}, []byte{1, 2, 3, 4, 5, 6})
	rd, err := NewIDXReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("NewIDXReader: %v", err)
	}
	_, _ = rd.Next() // first record consumes 4 bytes (ok)
	_, err = rd.Next()
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("short Next err = %v, want ErrIO", err)
	}
}

func TestMNISTLoaderBatching(t *testing.T) {
	t.Parallel()
	images := buildIDX([]int32{5, 2}, []byte{
		0, 255, 0, 255, 0,
		255, 128, 0, 128, 255,
	})
	labels := buildIDX([]int32{5}, []byte{0, 1, 2, 3, 4})
	imgR, err := NewIDXReader(bytes.NewReader(images))
	if err != nil {
		t.Fatalf("imgR: %v", err)
	}
	lblR, err := NewIDXReader(bytes.NewReader(labels))
	if err != nil {
		t.Fatalf("lblR: %v", err)
	}
	ldr, err := NewMNISTLoader[float32](imgR, lblR, 2, 255)
	if err != nil {
		t.Fatalf("NewMNISTLoader: %v", err)
	}
	if got, ok := ldr.Len(); !ok || got != 5 {
		t.Errorf("Len = (%d,%v), want (5,true)", got, ok)
	}
	ctx := context.Background()
	totalSamples := 0
	for {
		b, err := ldr.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		totalSamples += b.Len()
		for i, in := range b.Inputs {
			if len(in) != 2 {
				t.Errorf("input len = %d, want 2", len(in))
			}
			for _, v := range in {
				if v < 0 || v > 1 {
					t.Errorf("normalised pixel %v outside [0,1]", v)
				}
			}
			if len(b.Targets[i]) != 1 {
				t.Errorf("target len = %d, want 1", len(b.Targets[i]))
			}
		}
	}
	if totalSamples != 5 {
		t.Errorf("total samples = %d, want 5", totalSamples)
	}
}

func TestMNISTLoaderMismatch(t *testing.T) {
	t.Parallel()
	images := buildIDX([]int32{3, 2}, []byte{0, 0, 0, 0, 0, 0})
	labels := buildIDX([]int32{2}, []byte{0, 1})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	_, err := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	if err == nil || !errors.Is(err, utils.ErrMNISTRecordMismatch) {
		t.Errorf("err = %v, want ErrMNISTRecordMismatch", err)
	}
}

func TestMNISTLoaderArgValidation(t *testing.T) {
	t.Parallel()
	images := buildIDX([]int32{1, 1}, []byte{0})
	labels := buildIDX([]int32{1}, []byte{0})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	if _, err := NewMNISTLoader[float32](nil, lblR, 1, 255); err == nil {
		t.Error("nil images accepted")
	}
	if _, err := NewMNISTLoader[float32](imgR, lblR, 0, 255); err == nil {
		t.Error("zero batchSize accepted")
	}
	imgR2, _ := NewIDXReader(bytes.NewReader(images))
	lblR2, _ := NewIDXReader(bytes.NewReader(labels))
	if _, err := NewMNISTLoader[float32](imgR2, lblR2, 1, 0); err == nil {
		t.Error("zero norm accepted")
	}
}

func TestMNISTLoaderFromFiles(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	imgPath := filepath.Join(dir, "train.idx")
	lblPath := filepath.Join(dir, "labels.idx")
	if err := os.WriteFile(imgPath, buildIDX([]int32{2, 2}, []byte{0, 64, 128, 255}), 0o644); err != nil {
		t.Fatalf("WriteFile img: %v", err)
	}
	if err := os.WriteFile(lblPath, buildIDX([]int32{2}, []byte{0, 1}), 0o644); err != nil {
		t.Fatalf("WriteFile lbl: %v", err)
	}
	ldr, err := NewMNISTLoaderFiles[float32](imgPath, lblPath, 1, 255)
	if err != nil {
		t.Fatalf("NewMNISTLoaderFiles: %v", err)
	}
	defer func() {
		if err := ldr.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()
	ctx := context.Background()
	count := 0
	for {
		_, err := ldr.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		count++
	}
	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}
	// Reset should reopen the files and produce the same batches again.
	if err := ldr.Reset(ctx); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	count = 0
	for {
		_, err := ldr.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next post-Reset: %v", err)
		}
		count++
	}
	if count != 2 {
		t.Errorf("post-Reset count = %d, want 2", count)
	}
}

func TestMNISTLoaderFilesMissing(t *testing.T) {
	t.Parallel()
	_, err := NewMNISTLoaderFiles[float32]("/does/not/exist.idx", "/does/not/exist-lbl.idx", 1, 255)
	if err == nil || !errors.Is(err, utils.ErrIO) {
		t.Errorf("err = %v, want ErrIO", err)
	}
}

func TestMNISTLoaderContextCancelled(t *testing.T) {
	t.Parallel()
	images := buildIDX([]int32{1, 1}, []byte{0})
	labels := buildIDX([]int32{1}, []byte{0})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, _ := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ldr.Next(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("Next ctx-cancelled err = %v, want context.Canceled", err)
	}
	if err := ldr.Reset(ctx); !errors.Is(err, context.Canceled) {
		t.Errorf("Reset ctx-cancelled err = %v, want context.Canceled", err)
	}
}

func TestMNISTLoaderImageShapeFromHeader(t *testing.T) {
	t.Parallel()
	// 3-D IDX: N=2, H=4, W=4 — ImageShape should return (1, 4, 4).
	images := buildIDX3D([]int32{2, 4, 4}, make([]byte, 2*4*4))
	labels := buildIDX([]int32{2}, []byte{0, 1})
	imgR, err := NewIDXReader(bytes.NewReader(images))
	if err != nil {
		t.Fatalf("NewIDXReader images: %v", err)
	}
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, err := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	if err != nil {
		t.Fatalf("NewMNISTLoader: %v", err)
	}
	c, h, w := ldr.ImageShape()
	if c != 1 || h != 4 || w != 4 {
		t.Errorf("ImageShape() = (%d,%d,%d), want (1,4,4)", c, h, w)
	}
}

func TestMNISTLoaderImageShapeWithImageShape(t *testing.T) {
	t.Parallel()
	// Flat 1-D IDX: 3*8*8=192 bytes per record; explicit WithImageShape(3,8,8).
	images := buildIDX([]int32{2, 192}, make([]byte, 2*192))
	labels := buildIDX([]int32{2}, []byte{0, 1})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, _ := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	ldr.WithImageShape(3, 8, 8)
	c, h, w := ldr.ImageShape()
	if c != 3 || h != 8 || w != 8 {
		t.Errorf("WithImageShape(3,8,8): ImageShape() = (%d,%d,%d), want (3,8,8)", c, h, w)
	}
}

func TestMNISTLoaderImageShapeIsqrtFallback(t *testing.T) {
	t.Parallel()
	// Flat 1-D IDX with recLen=784 (28*28). ImageShape should infer (1,28,28).
	images := buildIDX([]int32{1, 784}, make([]byte, 784))
	labels := buildIDX([]int32{1}, []byte{3})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, _ := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	c, h, w := ldr.ImageShape()
	if c != 1 || h != 28 || w != 28 {
		t.Errorf("isqrt fallback: ImageShape() = (%d,%d,%d), want (1,28,28)", c, h, w)
	}
}

func TestMNISTLoaderImageShapeUnknown(t *testing.T) {
	t.Parallel()
	// Flat 1-D IDX with non-square recLen=100 (10*10 is 100 — actually this IS square).
	// Use 200 to ensure non-square result.
	images := buildIDX([]int32{1, 200}, make([]byte, 200))
	labels := buildIDX([]int32{1}, []byte{0})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, _ := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	c, h, w := ldr.ImageShape()
	if c != 0 || h != 0 || w != 0 {
		t.Errorf("non-square: ImageShape() = (%d,%d,%d), want (0,0,0)", c, h, w)
	}
}

func TestImageShaperInterface(t *testing.T) {
	t.Parallel()
	images := buildIDX([]int32{1, 784}, make([]byte, 784))
	labels := buildIDX([]int32{1}, []byte{0})
	imgR, _ := NewIDXReader(bytes.NewReader(images))
	lblR, _ := NewIDXReader(bytes.NewReader(labels))
	ldr, _ := NewMNISTLoader[float32](imgR, lblR, 1, 255)
	var _ ImageShaper = ldr // compile-time interface check
	if _, ok := any(ldr).(ImageShaper); !ok {
		t.Error("MNISTLoader does not implement ImageShaper")
	}
}

// buildIDX3D builds an in-memory 3-D IDX byte stream (N×H×W).
func buildIDX3D(dims []int32, data []byte) []byte {
	buf := &bytes.Buffer{}
	buf.WriteByte(0x00)
	buf.WriteByte(0x00)
	buf.WriteByte(idxDTypeUInt8)
	buf.WriteByte(byte(len(dims)))
	for _, d := range dims {
		_ = binary.Write(buf, binary.BigEndian, uint32(d))
	}
	buf.Write(data)
	return buf.Bytes()
}

func TestElementSize(t *testing.T) {
	t.Parallel()
	cases := []struct {
		dtype byte
		want  int
	}{
		{idxDTypeUInt8, 1}, {idxDTypeInt8, 1},
		{idxDTypeInt16, 2},
		{idxDTypeInt32, 4}, {idxDTypeFloat, 4},
		{idxDTypeDouble, 8},
		{0xFF, 1},
	}
	for _, tc := range cases {
		got := idxHeader{dtype: tc.dtype}.elementSize()
		if got != tc.want {
			t.Errorf("elementSize(%#x) = %d, want %d", tc.dtype, got, tc.want)
		}
	}
}
