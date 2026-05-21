package aimeta

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// Options controls the behaviour of [Check].
//
// AI-Meta:
//   - Purpose: Configuration for Check — controls tier enforcement and future resolver activation.
//   - Usage: aimeta.Check(".", aimeta.Options{IncludeInternal: true}).
//   - Related: [Check], [Tier].
//   - Stability: Stable.
type Options struct {
	// Resolve activates the symbol resolver (--resolve flag). Reserved for
	// Phase 16 — ignored in Phase 15.
	Resolve bool
	// IncludeInternal enables TIER violation checks for internal-tier symbols
	// (pkg/* packages other than pkg/nn). When false, TIER checks apply only
	// to public-API-tier symbols (pkg/nn, cmd/).
	IncludeInternal bool
}

// Check parses all non-test .go files in pkgPath, extracts AI-Meta blocks
// from exported identifiers, and returns any violations. It does not follow
// sub-directories.
//
// AI-Meta:
//   - Purpose: Lint all exported symbols in a package directory for AI-Meta block compliance.
//   - Usage: viols, err := aimeta.Check(".", aimeta.Options{IncludeInternal: true}).
//   - Errors: ErrIO-wrapped error if the directory cannot be parsed.
//   - Concurrency: Safe; creates a fresh token.FileSet per call.
//   - Related: [Options], [Violation], [ExtractBlock], [ParseBlock].
//   - Stability: Stable.
func Check(pkgPath string, opts Options) ([]Violation, error) {
	absPath, err := filepath.Abs(pkgPath)
	if err != nil {
		return nil, fmt.Errorf("aimeta.Check: resolve path %q: %w", pkgPath, err)
	}

	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, absPath, func(fi os.FileInfo) bool { //nolint:staticcheck
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("aimeta.Check: parse %q: %w", absPath, err)
	}

	tier := inferTier(absPath)
	var all []Violation
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			all = append(all, checkFile(fset, file, tier, opts)...)
		}
	}
	return all, nil
}

// inferTier determines the documentation tier from the absolute package path.
func inferTier(absPath string) Tier {
	clean := filepath.ToSlash(absPath)
	if strings.Contains(clean, "/pkg/nn") || strings.Contains(clean, "/cmd/") {
		return TierPublicAPI
	}
	return TierInternal
}

// checkFile inspects all exported declarations in a file for AI-Meta compliance.
func checkFile(fset *token.FileSet, file *ast.File, tier Tier, opts Options) []Violation {
	var violations []Violation
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Name.IsExported() {
				isMethod := d.Recv != nil
				violations = append(violations,
					checkDoc(fset, d.Doc, d.Name.Name, tier, opts, isMethod)...)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				violations = append(violations, checkGenSpec(fset, d, spec, tier, opts)...)
			}
		}
	}
	return violations
}

// checkGenSpec handles TypeSpec and ValueSpec inside a GenDecl.
func checkGenSpec(fset *token.FileSet, d *ast.GenDecl, spec ast.Spec, tier Tier, opts Options) []Violation {
	switch s := spec.(type) {
	case *ast.TypeSpec:
		if s.Name.IsExported() {
			doc := s.Doc
			if doc == nil {
				doc = d.Doc
			}
			return checkDoc(fset, doc, s.Name.Name, tier, opts, false)
		}
	case *ast.ValueSpec:
		var violations []Violation
		for _, name := range s.Names {
			if name.IsExported() {
				doc := s.Doc
				if doc == nil {
					doc = d.Doc
				}
				violations = append(violations, checkDoc(fset, doc, name.Name, tier, opts, false)...)
			}
		}
		return violations
	}
	return nil
}

// checkDoc validates the AI-Meta block (if any) in the doc comment for a
// symbol named sym. isMethod suppresses TIER checks — methods inherit tier
// requirements from their receiver type.
func checkDoc(fset *token.FileSet, doc *ast.CommentGroup, sym string, tier Tier, opts Options, isMethod bool) []Violation {
	block, ok := ExtractBlock(doc)
	if !ok {
		// No AI-Meta block — skip (no TIER violation for absent blocks in Phase 15).
		return nil
	}

	rawText := strings.Join(block.RawLines, "\n")
	parsed, violations := ParseBlock(rawText)

	// Compute position for violations from the first comment line.
	var pos token.Position
	if fset != nil && doc != nil && len(doc.List) > 0 {
		pos = fset.Position(doc.List[0].Slash)
	}
	for i := range violations {
		violations[i].Symbol = sym
		if !violations[i].Pos.IsValid() {
			violations[i].Pos = pos
		}
	}

	// LAST: non-blank lines follow the block in the comment group.
	if block.HasTrailingText {
		violations = append(violations, Violation{
			Pos:     pos,
			Symbol:  sym,
			Rule:    RuleLast,
			Message: fmt.Sprintf("%s: AI-Meta block must be the last content in the doc comment", sym),
		})
	}

	// TIER: check required fields (only when a block is present).
	if !isMethod {
		tierActive := tier == TierPublicAPI || opts.IncludeInternal
		if tierActive {
			for _, req := range TierRequiredFields[tier] {
				if _, present := parsed.Fields[req]; !present {
					violations = append(violations, Violation{
						Pos:    pos,
						Symbol: sym,
						Rule:   RuleTier,
						Message: fmt.Sprintf(
							"%s: missing required field %q for %s tier",
							sym, req, tierName(tier)),
					})
				}
			}
		}
	}

	return violations
}

func tierName(t Tier) string {
	if t == TierPublicAPI {
		return "public-API"
	}
	return "internal"
}
