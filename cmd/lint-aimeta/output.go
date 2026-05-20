package main

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/teratron/gonn/pkg/aimeta"
)

// printText writes one violation per line in the format:
//
//	file:line:col [RULE] symbol: message
func printText(w io.Writer, violations []aimeta.Violation) {
	for _, v := range violations {
		pos := v.Pos.String()
		if pos == "-" || pos == "" {
			pos = "<unknown>"
		}
		fmt.Fprintf(w, "%s [%s] %s: %s\n", pos, v.Rule, v.Symbol, v.Message)
	}
}

// violationJSON is the on-wire shape for JSON output.
type violationJSON struct {
	File    string `json:"file,omitempty"`
	Symbol  string `json:"symbol,omitempty"`
	Rule    string `json:"rule"`
	Message string `json:"message"`
	Line    int    `json:"line,omitempty"`
	Column  int    `json:"column,omitempty"`
}

// printJSON writes newline-delimited JSON records, one per violation.
func printJSON(w io.Writer, violations []aimeta.Violation) error {
	enc := json.NewEncoder(w)
	for _, v := range violations {
		rec := violationJSON{
			Symbol:  v.Symbol,
			Rule:    v.Rule,
			Message: v.Message,
		}
		if v.Pos.IsValid() {
			rec.File = v.Pos.Filename
			rec.Line = v.Pos.Line
			rec.Column = v.Pos.Column
		}
		if err := enc.Encode(rec); err != nil {
			return err
		}
	}
	return nil
}
