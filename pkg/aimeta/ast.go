package aimeta

import (
	"go/ast"
	"strings"
)

// Block holds the raw lines and, after ParseBlock is called, the parsed
// fields of an AI-Meta doc-comment block.
//
// AI-Meta:
//   - Purpose: Container for an extracted and/or parsed AI-Meta block from a doc comment.
//   - Related: [ExtractBlock], [ParseBlock], [Violation].
//   - Stability: Stable.
type Block struct {
	// Fields maps field names to their single-line values. Populated by
	// ParseBlock; nil if the block has not been parsed yet.
	Fields map[string]string
	// RawLines holds the raw block lines (stripped of the "// " prefix),
	// starting with "AI-Meta:". Populated by ExtractBlock.
	RawLines []string
	// LineCount is the total number of lines in the block including the
	// "AI-Meta:" label line.
	LineCount int
	// HasTrailingText is true when there are non-blank comment lines after
	// the block in the same comment group. Used by Check to emit LAST violations.
	HasTrailingText bool
}

// ExtractBlock locates the AI-Meta trailing block in cg and returns a Block
// with RawLines, LineCount, and HasTrailingText populated. Returns
// (Block{}, false) when no "AI-Meta:" label is found in the group.
//
// AI-Meta:
//   - Purpose: Locate and slice the AI-Meta block from an AST comment group; populate Block.RawLines.
//   - Usage: block, ok := aimeta.ExtractBlock(doc); if ok { block, viols := aimeta.ParseBlock(strings.Join(block.RawLines, "\n")) }.
//   - Related: [Block], [ParseBlock].
//   - Stability: Stable.
func ExtractBlock(cg *ast.CommentGroup) (Block, bool) {
	if cg == nil {
		return Block{}, false
	}

	lines := commentGroupLines(cg)

	startIdx := -1
	for i, line := range lines {
		if line == "AI-Meta:" {
			startIdx = i
			break
		}
	}
	if startIdx < 0 {
		return Block{}, false
	}

	// Collect lines from "AI-Meta:" until the first blank line or end.
	blockLines := []string{lines[startIdx]}
	endIdx := startIdx
	for i := startIdx + 1; i < len(lines); i++ {
		if lines[i] == "" {
			break
		}
		blockLines = append(blockLines, lines[i])
		endIdx = i
	}

	// Detect trailing non-blank lines after the block (LAST check).
	hasTrailing := false
	for i := endIdx + 1; i < len(lines); i++ {
		if lines[i] != "" {
			hasTrailing = true
			break
		}
	}

	return Block{
		RawLines:        blockLines,
		LineCount:       len(blockLines),
		HasTrailingText: hasTrailing,
	}, true
}

// commentGroupLines strips the "// " (or "//") prefix from each comment in
// the group and returns the resulting text lines.
func commentGroupLines(cg *ast.CommentGroup) []string {
	lines := make([]string, 0, len(cg.List))
	for _, c := range cg.List {
		text := c.Text
		switch {
		case strings.HasPrefix(text, "// "):
			lines = append(lines, text[3:])
		case text == "//":
			lines = append(lines, "")
		case strings.HasPrefix(text, "//"):
			// e.g. "//word" without space — strip "//"
			lines = append(lines, text[2:])
		}
	}
	return lines
}
