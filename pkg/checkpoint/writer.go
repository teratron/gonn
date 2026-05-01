// Package checkpoint — gzip cold-tier helper.
//
// Implements [l2-checkpointing-impl] §5.4 promotion step: hot snapshots
// stay as plain .json; older snapshots are gzipped in place to save
// disk while remaining readable by LoadLatest (which auto-detects the
// magic bytes).
package checkpoint

import (
	"compress/gzip"
	"os"

	"github.com/teratron/gonn/pkg/utils"
)

// gzipFile compresses src into src + ".gz" using compress/gzip and then
// removes src on success. On any failure the partial destination is
// removed and the caller receives the original src path with an
// ErrIO-wrapped error so logging stays useful.
func gzipFile(src string) (string, error) {
	raw, err := os.ReadFile(src)
	if err != nil {
		return src, utils.Wrap(utils.ErrIO, err, "gzipFile: read %q", src)
	}

	dst := src + ".gz"
	out, err := os.Create(dst)
	if err != nil {
		return src, utils.Wrap(utils.ErrIO, err, "gzipFile: create %q", dst)
	}
	gz := gzip.NewWriter(out)
	if _, err := gz.Write(raw); err != nil {
		_ = gz.Close()
		_ = out.Close()
		_ = os.Remove(dst)
		return src, utils.Wrap(utils.ErrIO, err, "gzipFile: write")
	}
	if err := gz.Close(); err != nil {
		_ = out.Close()
		_ = os.Remove(dst)
		return src, utils.Wrap(utils.ErrIO, err, "gzipFile: close gz")
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(dst)
		return src, utils.Wrap(utils.ErrIO, err, "gzipFile: close out")
	}
	if err := os.Remove(src); err != nil {
		return dst, utils.Wrap(utils.ErrIO, err, "gzipFile: remove src")
	}
	return dst, nil
}
