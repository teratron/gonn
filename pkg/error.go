package pkg

import "errors"

var (
	// ErrInit - initialization error.
	ErrInit = errors.New("initialization error")

	// ErrNotRecognized - not recognized.
	ErrNotRecognized = errors.New("not recognized")

	// ErrMissingType - type is missing.
	ErrMissingType = errors.New("type is missing")

	// ErrNoInput - no input data.
	ErrNoInput = errors.New("no input data")

	// ErrNoTarget - no target data.
	ErrNoTarget = errors.New("no target data")

	// ErrNoArgs - arguments is missing.
	ErrNoArgs = errors.New("no arguments")
)
