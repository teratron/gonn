package utils

import (
	"log/slog"
	"os"
)

var Logger *slog.Logger

func init() {
	Logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
}
