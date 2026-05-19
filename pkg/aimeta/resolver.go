package aimeta

import (
	"bufio"
	"bytes"
	"fmt"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// RESOLVE rule codes — emitted for each fix the resolver applies.
// These differ from the violation rule codes: a RESOLVE-* entry means the
// violation was automatically repaired, not that it persists.
const (
	// RuleResolveLabel is emitted when the "AI-Meta:" label casing is corrected.
	RuleResolveLabel = "RESOLVE-LABEL"
	// RuleResolveIndent is emitted when a field line's indentation is corrected.
	RuleResolveIndent = "RESOLVE-INDENT"
	// RuleResolveLast is emitted when a missing terminal newline is appended.
	RuleResolveLast = "RESOLVE-LAST"
)

// Resolve reads every non-test .go file under pkgPath (non-recursive), applies
// mechanical fixes for LABEL, INDENT, and LAST violations, and writes back any
// modified files. Returns one Violation record per fix applied (Rule =
// RESOLVE-*). Unfixable rules (VOCAB, TIER, MULTI, ENUM, ARTIFACT) are skipped.
//
// AI-Meta:
//   - Purpose: Auto-fix LABEL/INDENT/LAST AI-Meta violations in-place across a package directory.
//   - Usage: fixes, err := aimeta.Resolve("./pkg/foo", aimeta.Options{}).
//   - Errors: ErrIO-wrapped error if a file cannot be read or written.
//   - Concurrency: NotSafe; writes files on disk.
//   - Related: [Check], [Options], [RuleResolveLabel], [RuleResolveIndent], [RuleResolveLast].
//   - Stability: Stable.
func Resolve(pkgPath string, opts Options) ([]Violation, error) {
	absPath, err := filepath.Abs(pkgPath)
	if err != nil {
		return nil, fmt.Errorf("aimeta.Resolve: resolve path %q: %w", pkgPath, err)
	}

	entries, err := os.ReadDir(absPath)
	if err != nil {
		return nil, fmt.Errorf("aimeta.Resolve: read dir %q: %w", absPath, err)
	}

	var all []Violation
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		filePath := filepath.Join(absPath, name)
		fixes, err := resolveFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("aimeta.Resolve: file %q: %w", filePath, err)
		}
		all = append(all, fixes...)
	}
	return all, nil
}

// resolveFile applies mechanical fixes to a single Go source file. Returns the
// list of applied fixes. Writes the file back only when at least one fix was
// applied.
func resolveFile(path string) ([]Violation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	fixed, fixes := applyFixes(path, data)

	if len(fixes) > 0 {
		if err := os.WriteFile(path, fixed, 0o644); err != nil {
			return nil, err
		}
	}
	return fixes, nil
}

// applyFixes runs each fix pass over data and returns the (possibly modified)
// content and a list of Violation records for each fix applied.
func applyFixes(path string, data []byte) ([]byte, []Violation) {
	var fixes []Violation
	data, labelFixes := fixLabel(path, data)
	fixes = append(fixes, labelFixes...)
	data, indentFixes := fixIndent(path, data)
	fixes = append(fixes, indentFixes...)
	data, lastFixes := fixLast(path, data)
	fixes = append(fixes, lastFixes...)
	return data, fixes
}

// fixLabel corrects the AI-Meta label casing: any line containing "// AI-"
// followed by [Mm]eta: (wrong case) is normalised to "// AI-Meta:".
func fixLabel(path string, data []byte) ([]byte, []Violation) {
	var out bytes.Buffer
	var fixes []Violation
	lineNum := 0

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if fixed, ok := fixLabelLine(line); ok {
			fixes = append(fixes, Violation{
				Pos:     posAt(path, lineNum),
				Rule:    RuleResolveLabel,
				Message: fmt.Sprintf("line %d: corrected AI-Meta label casing", lineNum),
			})
			out.WriteString(fixed)
		} else {
			out.WriteString(line)
		}
		out.WriteByte('\n')
	}
	if len(fixes) == 0 {
		return data, nil
	}
	return out.Bytes(), fixes
}

// fixLabelLine returns a corrected line and true when the line contains a
// malformed AI-Meta label (wrong casing). Matches "// AI-meta:", "// AI-META:",
// "// ai-meta:", etc. but NOT "// AI-Meta:" (already correct).
func fixLabelLine(line string) (string, bool) {
	trimmed := strings.TrimLeft(line, " \t")
	if !strings.HasPrefix(trimmed, "//") {
		return "", false
	}
	after := strings.TrimPrefix(trimmed, "//")
	after = strings.TrimLeft(after, " ")

	low := strings.ToLower(after)
	if !strings.HasPrefix(low, "ai-meta:") {
		return "", false
	}
	// Already correct?
	if strings.HasPrefix(after, "AI-Meta:") {
		return "", false
	}
	// Reconstruct: preserve leading spaces from original line.
	indent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
	rest := strings.TrimPrefix(after, after[:8]) // strip the wrong-case "ai-meta:" portion
	return indent + "// AI-Meta:" + rest, true
}

// fixIndent corrects INDENT violations: within an AI-Meta block, field lines
// that start with "// -" (missing indentation) or "//  - " (one space) are
// normalised to "//   - " (three spaces = two space indent + "- ").
func fixIndent(path string, data []byte) ([]byte, []Violation) {
	var out bytes.Buffer
	var fixes []Violation
	lineNum := 0
	inBlock := false

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		// Detect entry and exit of AI-Meta block.
		stripped := strings.TrimLeft(line, " \t")
		if strings.HasPrefix(stripped, "// AI-Meta:") {
			inBlock = true
			out.WriteString(line)
			out.WriteByte('\n')
			continue
		}
		// Block ends at blank line or non-comment line.
		if inBlock && (line == "" || !strings.HasPrefix(stripped, "//")) {
			inBlock = false
		}

		if inBlock {
			if fixed, ok := fixIndentLine(line); ok {
				fixes = append(fixes, Violation{
					Pos:     posAt(path, lineNum),
					Rule:    RuleResolveIndent,
					Message: fmt.Sprintf("line %d: corrected AI-Meta field indentation", lineNum),
				})
				out.WriteString(fixed)
			} else {
				out.WriteString(line)
			}
		} else {
			out.WriteString(line)
		}
		out.WriteByte('\n')
	}
	if len(fixes) == 0 {
		return data, nil
	}
	return out.Bytes(), fixes
}

// fixIndentLine corrects a comment field line that has wrong indentation.
// Correct form: "//   - Field: value" (two leading spaces after "// ").
// Fixes "// - Field:" (one space) and "//-" (no space).
func fixIndentLine(line string) (string, bool) {
	indent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
	rest := strings.TrimLeft(line, " \t")

	// Must start with "//"
	if !strings.HasPrefix(rest, "//") {
		return "", false
	}
	after := rest[2:] // strip "//"

	// Already correct: "   - " (three spaces + "- ") → "  - " stripped
	if strings.HasPrefix(after, "   - ") {
		return "", false
	}

	// Fixable patterns: "- " or " - " or "  - " (wrong number of spaces before "- ").
	trimmed := strings.TrimLeft(after, " ")
	if !strings.HasPrefix(trimmed, "- ") {
		return "", false
	}
	return indent + "//   " + trimmed, true
}

// fixLast appends a terminal newline when the file doesn't end with one.
func fixLast(path string, data []byte) ([]byte, []Violation) {
	if len(data) == 0 || data[len(data)-1] == '\n' {
		return data, nil
	}
	// Count total lines (the terminal newline goes on last line).
	lineCount := bytes.Count(data, []byte{'\n'}) + 1
	fix := Violation{
		Pos:     posAt(path, lineCount),
		Rule:    RuleResolveLast,
		Message: fmt.Sprintf("appended missing terminal newline at line %d", lineCount),
	}
	return append(data, '\n'), []Violation{fix}
}

// posAt constructs a token.Position for resolver fix records.
func posAt(path string, line int) token.Position {
	return token.Position{Filename: path, Line: line}
}
