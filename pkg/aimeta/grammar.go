package aimeta

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// artifactPattern matches SDD process-artifact references that must not appear
// in AI-Meta blocks: invariant numbers (INV-N) and .design/ path prefixes.
var artifactPattern = regexp.MustCompile(`INV-\d+|\.design/`)

// ParseBlock parses raw AI-Meta block text (as returned by
// strings.Join(block.RawLines, "\n")) and returns the structured Block with
// any grammar violations. Checks: LABEL, INDENT, CAP, VOCAB, MULTI, ENUM,
// ARTIFACT. Does NOT check TIER or LAST — those require caller context.
//
// AI-Meta:
//   - Purpose: Parse raw AI-Meta block text into structured fields and collect grammar violations.
//   - Usage: parsed, viols := aimeta.ParseBlock(strings.Join(block.RawLines, "\n")).
//   - Related: [ExtractBlock], [Block], [Violation].
//   - Stability: Stable.
func ParseBlock(rawText string) (Block, []Violation) {
	lines := strings.Split(rawText, "\n")
	var violations []Violation

	if len(lines) == 0 || lines[0] != "AI-Meta:" {
		violations = append(violations, Violation{
			Rule:    RuleLabel,
			Message: `AI-Meta block must start with the literal label "AI-Meta:"`,
		})
		return Block{}, violations
	}

	// CAP: total lines (including label) must not exceed 12.
	if len(lines) > 12 {
		violations = append(violations, Violation{
			Rule:    RuleCap,
			Message: fmt.Sprintf("AI-Meta block has %d lines; maximum is 12 (including label)", len(lines)),
		})
	}

	fields := make(map[string]string)

	for i := 1; i < len(lines); i++ {
		line := lines[i]

		// MULTI: nested bullet — 4+ leading spaces (indented continuation line).
		if strings.HasPrefix(line, "    ") {
			violations = append(violations, Violation{
				Rule: RuleMulti,
				Message: fmt.Sprintf(
					"line %d: multi-line value detected — use comma-separated list on a single line", i+1),
			})
			continue
		}

		// INDENT: must start with "  - ".
		if !strings.HasPrefix(line, "  - ") {
			violations = append(violations, Violation{
				Rule:    RuleIndent,
				Message: fmt.Sprintf("line %d: expected \"  - <Field>: <value>\" indentation, got %q", i+1, line),
			})
			continue
		}

		rest := line[4:] // strip "  - "

		// Find ": " separator. Handle "Field:" with no value.
		var name, value string
		if n, v, found := strings.Cut(rest, ": "); found {
			name, value = n, v
		} else {
			name = strings.TrimSuffix(rest, ":")
		}

		// VOCAB: field name must be in AllowedFields.
		if !isAllowedField(name) {
			violations = append(violations, Violation{
				Rule: RuleVocab,
				Message: fmt.Sprintf(
					"line %d: unknown field %q; allowed: %s",
					i+1, name, strings.Join(AllowedFields, ", ")),
			})
			continue
		}

		fields[name] = value

		// ENUM: Stability and Concurrency have closed value sets.
		if v := checkEnum(name, value, i+1); v != nil {
			violations = append(violations, *v)
		}

		// ARTIFACT: no SDD process-artifact references inside the block.
		if v := checkArtifact(name, value, i+1); v != nil {
			violations = append(violations, *v)
		}
	}

	return Block{Fields: fields, LineCount: len(lines)}, violations
}

func isAllowedField(name string) bool {
	return slices.Contains(AllowedFields, name)
}

func checkEnum(name, value string, lineNum int) *Violation {
	// Strip trailing punctuation — doc-comment convention allows "Stable." etc.
	norm := strings.TrimRight(value, ".,;:")

	switch name {
	case "Stability":
		if !slices.Contains(StabilityEnum, norm) {
			return &Violation{
				Rule: RuleEnum,
				Message: fmt.Sprintf(
					"line %d: Stability value %q is not in {%s}",
					lineNum, value, strings.Join(StabilityEnum, ", ")),
			}
		}

	case "Concurrency":
		// Value may have "; <clarifier>" suffix — compare only the base.
		base, _, _ := strings.Cut(norm, ";")
		base = strings.TrimSpace(base)
		if !slices.Contains(ConcurrencyEnum, base) {
			return &Violation{
				Rule: RuleEnum,
				Message: fmt.Sprintf(
					"line %d: Concurrency value %q is not in {%s}",
					lineNum, value, strings.Join(ConcurrencyEnum, ", ")),
			}
		}
	}
	return nil
}

func checkArtifact(name, value string, lineNum int) *Violation {
	if artifactPattern.MatchString(value) {
		return &Violation{
			Rule: RuleArtifact,
			Message: fmt.Sprintf(
				"line %d: field %q references an SDD process artifact: %q",
				lineNum, name, value),
		}
	}
	return nil
}
