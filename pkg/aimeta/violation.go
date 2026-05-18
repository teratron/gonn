// Package aimeta — AI-Meta doc-comment block extractor and grammar linter.
//
// Implements [l2-aimeta-linter] §5.1–§5.4. Provides a stdlib-only AST walker
// that extracts and validates the AI-Meta trailing block defined by
// [l2-ai-doc-metadata] §3–§6. Import as a library for per-package
// TestAIMetaCompliance hooks; use [cmd/lint-aimeta] for project-wide linting.
//
// AI-Meta:
//   - Purpose: Stdlib-only linter for AI-Meta doc-comment blocks; enforces grammar and vocabulary rules.
//   - Usage: aimeta.Check("./pkg/foo", aimeta.Options{IncludeInternal: true}) returns violations.
//   - Concurrency: Safe; Check creates a fresh FileSet per call.
//   - Related: [Check], [ExtractBlock], [ParseBlock], [Violation].
//   - Stability: Stable.
package aimeta

import "go/token"

// Violation describes a single rule failure found in an AI-Meta block.
//
// AI-Meta:
//   - Purpose: Carry the position, symbol name, rule code, and human message for one AI-Meta violation.
//   - Usage: for _, v := range violations { t.Errorf("%s:%d [%s] %s", v.Pos.Filename, v.Pos.Line, v.Rule, v.Message) }.
//   - Related: [Check], [ParseBlock].
//   - Stability: Stable.
type Violation struct {
	// Pos is the source position of the offending doc comment or line.
	// Zero value when position information is unavailable.
	Pos token.Position
	// Symbol is the name of the exported identifier whose doc comment contains
	// the violation. Empty when the violation is file-level.
	Symbol string
	// Rule is one of the RuleXxx constants.
	Rule string
	// Message is a human-readable description of the violation.
	Message string
}

// Rule codes for AI-Meta violations.
const (
	// RuleLabel: the "AI-Meta:" label is missing or malformed (wrong casing, extra space).
	RuleLabel = "LABEL"
	// RuleIndent: a field line does not follow the "  - <Field>: <value>" indentation pattern.
	RuleIndent = "INDENT"
	// RuleCap: the block (including the label line) exceeds the 12-line maximum.
	RuleCap = "CAP"
	// RuleLast: the AI-Meta block is not the last content in the doc comment.
	RuleLast = "LAST"
	// RuleVocab: a field name is not in the closed vocabulary.
	RuleVocab = "VOCAB"
	// RuleTier: a field required by the symbol's documentation tier is absent.
	RuleTier = "TIER"
	// RuleMulti: a field value spans multiple lines (nested bullet list).
	RuleMulti = "MULTI"
	// RuleEnum: a Stability or Concurrency value is not in the closed enum set.
	RuleEnum = "ENUM"
	// RuleArtifact: the block references an SDD process artifact (INV-N, .design/ path).
	RuleArtifact = "ARTIFACT"
)
