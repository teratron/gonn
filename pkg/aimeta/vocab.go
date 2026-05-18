package aimeta

// Tier classifies a symbol's documentation tier, which determines the set
// of required AI-Meta fields.
//
// AI-Meta:
//   - Purpose: Enum classifying symbols into documentation tiers for required-field enforcement.
//   - Related: [TierRequiredFields], [Check].
//   - Stability: Stable.
type Tier uint8

const (
	// TierPublicAPI covers pkg/nn and cmd/ — the library's public surface.
	TierPublicAPI Tier = 0
	// TierInternal covers other pkg/* packages — internal but exported.
	TierInternal Tier = 1
)

// AllowedFields is the closed vocabulary of permitted AI-Meta field names,
// in the order specified by l2-ai-doc-metadata §4.1.
//
// AI-Meta:
//   - Purpose: Closed list of valid AI-Meta field names; used by ParseBlock for VOCAB enforcement.
//   - Related: [TierRequiredFields], [ParseBlock].
//   - Stability: Stable.
var AllowedFields = []string{
	"Purpose",
	"Usage",
	"Lifecycle",
	"Concurrency",
	"Errors",
	"Related",
	"Constraints",
	"Implementations",
	"Stability",
}

// StabilityEnum holds the valid values for the Stability field.
//
// AI-Meta:
//   - Purpose: Closed enum for the Stability field; used by ParseBlock for ENUM enforcement.
//   - Related: [ConcurrencyEnum], [ParseBlock].
//   - Stability: Stable.
var StabilityEnum = []string{"Stable", "Experimental", "Deprecated", "Internal"}

// ConcurrencyEnum holds the valid base values for the Concurrency field.
// The value may carry a "; <clarifier>" suffix which is stripped before comparison.
//
// AI-Meta:
//   - Purpose: Closed enum for the Concurrency field base value; used by ParseBlock for ENUM enforcement.
//   - Related: [StabilityEnum], [ParseBlock].
//   - Stability: Stable.
var ConcurrencyEnum = []string{"Safe", "ReadSafe", "SingleGoroutine", "NotSafe"}

// TierRequiredFields maps each Tier to the field names that must be present
// in the AI-Meta block. Tier checks are applied by [Check] when the block
// exists; symbols with no AI-Meta block at all are not flagged.
//
// AI-Meta:
//   - Purpose: Map from Tier to required field names; used by Check for TIER violation detection.
//   - Related: [Tier], [TierPublicAPI], [TierInternal], [Check].
//   - Stability: Stable.
var TierRequiredFields = map[Tier][]string{
	TierPublicAPI: {"Purpose", "Usage", "Concurrency", "Related", "Stability"},
	TierInternal:  {"Purpose"},
}
