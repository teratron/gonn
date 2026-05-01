// Package checkpoint — snapshot loader and schema migration.
//
// Implements [l2-checkpointing-impl] §5.1 reader: detect plain vs gzip
// by magic bytes, dispatch to migrate() when the schema_version differs
// from the library's, and return the deserialised Snapshot. Truncated
// or otherwise malformed files surface as ErrIntegrity (CHK-1) so the
// training loop can route via errors.Is.
package checkpoint

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/teratron/gonn/pkg/utils"
)

// gzipMagic is the first two bytes of the standard gzip header. Used to
// auto-detect compressed snapshots without relying on the .gz suffix.
var gzipMagic = []byte{0x1f, 0x8b}

// LoadLatest returns the most recent snapshot in dir along with the path
// it was loaded from. "Most recent" is defined by the Iter encoded in
// the filename — ties are broken by Timestamp. Both .json and .json.gz
// snapshots are considered.
func LoadLatest[T utils.Float](dir string) (Snapshot[T], string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return Snapshot[T]{}, "", utils.Wrap(utils.ErrIO, err, "LoadLatest: readdir %q", dir)
	}
	candidates := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, "snap-") &&
			(strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".json.gz")) {
			candidates = append(candidates, name)
		}
	}
	if len(candidates) == 0 {
		return Snapshot[T]{}, "", utils.Newf(utils.ErrIO,
			"LoadLatest: no snapshots in %q", dir)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(candidates)))
	target := filepath.Join(dir, candidates[0])

	snap, err := loadFile[T](target)
	if err != nil {
		return Snapshot[T]{}, target, err
	}
	return snap, target, nil
}

// loadFile reads a single snapshot file, transparently handling gzip
// compression. The function is exported only through LoadLatest to keep
// the public surface narrow.
func loadFile[T utils.Float](path string) (Snapshot[T], error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Snapshot[T]{}, utils.Wrap(utils.ErrIO, err, "loadFile: read %q", path)
	}
	body, err := maybeDecompress(raw)
	if err != nil {
		return Snapshot[T]{}, err
	}
	var snap Snapshot[T]
	if err := json.Unmarshal(body, &snap); err != nil {
		return Snapshot[T]{}, utils.Wrap(utils.ErrIntegrity, err,
			"loadFile: decode %q (truncated?)", path)
	}
	if snap.SchemaVersion != SchemaVersion {
		migrated, err := migrate[T](snap.SchemaVersion, SchemaVersion, body)
		if err != nil {
			return snap, err
		}
		snap = migrated
	}
	return snap, nil
}

// maybeDecompress returns the body unchanged when raw is not gzipped;
// otherwise it streams the data through compress/gzip and returns the
// decompressed payload.
func maybeDecompress(raw []byte) ([]byte, error) {
	if len(raw) < 2 || !bytes.Equal(raw[:2], gzipMagic) {
		return raw, nil
	}
	gz, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		return nil, utils.Wrap(utils.ErrIntegrity, err, "maybeDecompress: gzip header")
	}
	defer gz.Close()
	body, err := io.ReadAll(gz)
	if err != nil {
		return nil, utils.Wrap(utils.ErrIntegrity, err, "maybeDecompress: gzip body")
	}
	return body, nil
}

// migrate is the (from, to) → migrator dispatcher invoked by loadFile
// when on-disk schema_version differs from SchemaVersion. v1.0 is the
// only supported schema today; future bumps register new entries here.
// Unknown source versions surface as ErrUserConfig per CHK-4. The raw
// payload is accepted so future migrators can pre-decode into a
// version-specific struct before re-encoding into Snapshot[T].
func migrate[T utils.Float](from, to string, raw []byte) (Snapshot[T], error) {
	_ = raw
	return Snapshot[T]{}, utils.Newf(utils.ErrUserConfig,
		"checkpoint: no migrator from %q to %q", from, to)
}
