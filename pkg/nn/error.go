package nn

import "errors"

var (
	ErrInit          = errors.New("initialization error")
	ErrNotRecognized = errors.New("not recognized")
	ErrMissingType   = errors.New("type is missing")
	ErrNoInput       = errors.New("no input data")
	ErrNoTarget      = errors.New("no target data")
	ErrEmpty         = errors.New("empty")
	ErrNoFile        = errors.New("file is missing")
)
