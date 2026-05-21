// Package lintaimeta implements the AI-Meta doc-comment convention checker.
//
// It parses Go source files with go/parser and inspects every exported
// declaration for a trailing AI-Meta: block. Violations are collected as
// [Finding] values and returned to the caller.
//
// Entry points: [LintDir] (directory), [LintFile] (single AST file),
// [ParseBlock] (raw comment lines, for unit testing).
package lintaimeta

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Tier classifies a package by its AI-Meta documentation obligation.
type Tier int

const (
	// TierPublicAPI requires a full AI-Meta block on every exported symbol.
	// Applies to: pkg/nn and cmd/.
	TierPublicAPI Tier = iota

	// TierInternal requires Purpose on every exported symbol that has a block;
	// it does not require that a block be present at all.
	// Applies to: all other pkg/* packages.
	TierInternal

	// TierSkip suppresses all checking — used for test files and unexported symbols.
	TierSkip
)

// Finding is one AI-Meta rule violation reported by the linter.
type Finding struct {
	File    string
	Symbol  string
	Message string
	Line    int
}

func (f Finding) String() string {
	return fmt.Sprintf("%s:%d: %s: %s", f.File, f.Line, f.Symbol, f.Message)
}

// DeterminePackageTier maps a directory path (slash-separated, relative to the
// module root) to a Tier. Both forward and backward slashes are normalised.
func DeterminePackageTier(pkgPath string) Tier {
	p := filepath.ToSlash(pkgPath)
	if strings.Contains(p, "pkg/nn") || strings.HasPrefix(p, "cmd/") {
		return TierPublicAPI
	}
	if strings.Contains(p, "pkg/") {
		return TierInternal
	}
	return TierSkip
}

// Block holds the parsed contents of one AI-Meta: section.
type Block struct {
	Fields map[string]string // closed-vocabulary field name → trimmed value
	Lines  int               // total lines, including the "AI-Meta:" header
}

var (
	closedFields = map[string]bool{
		"Purpose": true, "Usage": true, "Lifecycle": true,
		"Concurrency": true, "Errors": true, "Related": true,
		"Constraints": true, "Implementations": true, "Stability": true,
	}

	validStability   = map[string]bool{"Stable": true, "Experimental": true, "Deprecated": true, "Internal": true}
	validConcurrency = [...]string{"Safe", "ReadSafe", "SingleGoroutine", "NotSafe"}

	// sddArtifactRe matches process-internal references forbidden in AI-Meta blocks.
	sddArtifactRe = regexp.MustCompile(`(INV-\d+|\.design/)`)
	// cRuleRe matches rule-number references like C25, C31.
	cRuleRe = regexp.MustCompile(`\bC\d{2,}\b`)
	// fieldLineRe matches "  - FieldName: value" (exactly two leading spaces).
	fieldLineRe = regexp.MustCompile(`^  - (\w+): (.+)$`)
)

// ParseBlock scans stripped comment lines (the "// " prefix already removed)
// for an AI-Meta: block. It returns the parsed block and a slice of violation
// messages. A nil block means no AI-Meta: section was found (not a violation
// by itself — tier rules decide whether that is required).
func ParseBlock(lines []string) (*Block, []string) {
	startIdx := -1
	for i, line := range lines {
		if strings.TrimSpace(line) == "AI-Meta:" {
			startIdx = i
			break
		}
	}
	if startIdx == -1 {
		return nil, nil
	}

	var violations []string

	// The line immediately before the block header must be blank.
	if startIdx > 0 && strings.TrimSpace(lines[startIdx-1]) != "" {
		violations = append(violations, "AI-Meta: block must be preceded by a blank comment line")
	}

	blockLines := lines[startIdx:]

	if len(blockLines) > 12 {
		violations = append(violations, fmt.Sprintf("AI-Meta: block exceeds 12-line cap (%d lines)", len(blockLines)))
	}

	block := &Block{Fields: make(map[string]string), Lines: len(blockLines)}

	for _, line := range blockLines[1:] {
		if strings.TrimSpace(line) == "" {
			// Blank lines inside the block signal prose after the block — violation.
			violations = append(violations, "AI-Meta: unexpected blank line inside block (block must be last)")
			continue
		}
		m := fieldLineRe.FindStringSubmatch(line)
		if m == nil {
			violations = append(violations, fmt.Sprintf("AI-Meta: malformed line %q (expected '  - Field: value')", line))
			continue
		}
		name, value := m[1], strings.TrimSpace(m[2])

		if !closedFields[name] {
			violations = append(violations, fmt.Sprintf("AI-Meta: unknown field %q", name))
		}
		if _, dup := block.Fields[name]; dup {
			violations = append(violations, fmt.Sprintf("AI-Meta: duplicate field %q", name))
		}
		block.Fields[name] = value

		if name == "Stability" && !validStability[value] {
			violations = append(violations, fmt.Sprintf("AI-Meta: Stability %q not in {Stable, Experimental, Deprecated, Internal}", value))
		}
		if name == "Concurrency" {
			if !hasValidConcurrencyPrefix(value) {
				violations = append(violations, fmt.Sprintf("AI-Meta: Concurrency %q must start with Safe|ReadSafe|SingleGoroutine|NotSafe", value))
			}
		}
		if sddArtifactRe.MatchString(value) || cRuleRe.MatchString(value) {
			violations = append(violations, fmt.Sprintf("AI-Meta: field %q contains a forbidden SDD artifact reference", name))
		}
	}

	return block, violations
}

func hasValidConcurrencyPrefix(value string) bool {
	for _, v := range validConcurrency {
		if strings.HasPrefix(value, v) {
			return true
		}
	}
	return false
}

// stripCommentLines converts a *ast.CommentGroup to a slice of plain text
// lines, removing the leading "// " (or "//") prefix from each comment.
func stripCommentLines(cg *ast.CommentGroup) []string {
	if cg == nil {
		return nil
	}
	out := make([]string, len(cg.List))
	for i, c := range cg.List {
		text := c.Text
		switch {
		case strings.HasPrefix(text, "// "):
			text = text[3:]
		case text == "//":
			text = ""
		case strings.HasPrefix(text, "//"):
			text = text[2:]
		}
		out[i] = text
	}
	return out
}

// declKind classifies an AST node's declaration kind for field-matrix lookups.
type declKind int

const (
	kindFunc      declKind = iota
	kindType               // struct, alias, or other non-interface type
	kindInterface          // interface type
	kindConst
	kindVar
)

// returnsError reports whether a FuncDecl has "error" anywhere in its result list.
func returnsError(fn *ast.FuncDecl) bool {
	if fn.Type == nil || fn.Type.Results == nil {
		return false
	}
	for _, field := range fn.Type.Results.List {
		if id, ok := field.Type.(*ast.Ident); ok && id.Name == "error" {
			return true
		}
	}
	return false
}

// checkDecl checks one exported symbol against the AI-Meta convention rules.
func checkDecl(
	name string,
	kind declKind,
	doc *ast.CommentGroup,
	hasErr bool,
	tier Tier,
	fset *token.FileSet,
) []Finding {
	if tier == TierSkip || doc == nil {
		return nil
	}

	lines := stripCommentLines(doc)
	block, parseViolations := ParseBlock(lines)

	pos := fset.Position(doc.Pos())
	var findings []Finding
	add := func(msg string) {
		findings = append(findings, Finding{File: pos.Filename, Line: pos.Line, Symbol: name, Message: msg})
	}

	for _, v := range parseViolations {
		add(v)
	}

	if block == nil {
		if tier == TierPublicAPI {
			add("missing AI-Meta: block (required for public API symbols)")
		}
		return findings
	}

	// Purpose is required whenever a block is present (any tier).
	if _, ok := block.Fields["Purpose"]; !ok {
		add("AI-Meta: missing required field 'Purpose'")
	}

	if tier == TierPublicAPI {
		if _, ok := block.Fields["Related"]; !ok {
			add("AI-Meta: missing required field 'Related'")
		}
		if _, ok := block.Fields["Stability"]; !ok {
			add("AI-Meta: missing required field 'Stability'")
		}
	}

	if hasErr {
		if _, ok := block.Fields["Errors"]; !ok {
			add("AI-Meta: missing required field 'Errors' (function returns error)")
		}
	}

	if kind == kindInterface {
		if _, ok := block.Fields["Implementations"]; !ok {
			add("AI-Meta: missing required field 'Implementations' (interface declaration)")
		}
	}

	return findings
}

// LintFile runs the linter against a single parsed Go source file.
func LintFile(fset *token.FileSet, file *ast.File, tier Tier) []Finding {
	var findings []Finding

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if !d.Name.IsExported() {
				continue
			}
			findings = append(findings, checkDecl(d.Name.Name, kindFunc, d.Doc, returnsError(d), tier, fset)...)

		case *ast.GenDecl:
			findings = append(findings, lintGenDecl(d, tier, fset)...)
		}
	}

	return findings
}

// lintGenDecl handles type, const, and var group declarations.
func lintGenDecl(d *ast.GenDecl, tier Tier, fset *token.FileSet) []Finding {
	var findings []Finding

	// Track whether we've already reported on the group-level doc.
	groupChecked := false

	for _, spec := range d.Specs {
		switch s := spec.(type) {
		case *ast.TypeSpec:
			if !s.Name.IsExported() {
				continue
			}
			doc := s.Doc
			if doc == nil {
				doc = d.Doc
			}
			kind := kindType
			if _, ok := s.Type.(*ast.InterfaceType); ok {
				kind = kindInterface
			}
			findings = append(findings, checkDecl(s.Name.Name, kind, doc, false, tier, fset)...)

		case *ast.ValueSpec:
			// For const/var groups, the group doc applies to all members.
			// Check it once using the first exported name.
			if d.Doc != nil && !groupChecked {
				for _, nm := range s.Names {
					if nm.IsExported() {
						findings = append(findings, checkDecl(nm.Name, constOrVar(d), nil, false, tier, fset)...)
						findings = append(findings, checkDecl(nm.Name, constOrVar(d), d.Doc, false, tier, fset)...)
						groupChecked = true
						break
					}
				}
				if groupChecked {
					// Skip per-spec check when group doc owns the block.
					continue
				}
			}
			// Individual spec with its own doc comment.
			if s.Doc != nil {
				for _, nm := range s.Names {
					if nm.IsExported() {
						findings = append(findings, checkDecl(nm.Name, constOrVar(d), s.Doc, false, tier, fset)...)
						break
					}
				}
			}
		}
	}

	return findings
}

func constOrVar(d *ast.GenDecl) declKind {
	if d.Tok.String() == "const" {
		return kindConst
	}
	return kindVar
}

// LintDir parses all non-test .go files in dir and returns every finding.
func LintDir(dir string, tier Tier) ([]Finding, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool { //nolint:staticcheck
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("lintaimeta: parse %q: %w", dir, err)
	}

	var findings []Finding
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			findings = append(findings, LintFile(fset, file, tier)...)
		}
	}
	return findings, nil
}
