---
phase: 13
name: "Convolutional 2-D Foundation"
status: Done
subsystem: ".design/main/specifications/ (L2 authoring); future pkg/layer/conv/, pkg/dataset/, examples/"
requires:
  - "Phase 11 ✓ (Conv1D Track B)"
  - "Phase 12 ✓ (Meta-Learning Hooks)"
  - "l1-conv-2d-layers Stable v0.2.0"
  - "sibling l1-conv-layers Stable v1.0.0"
provides:
  - "l1-conv-2d-layers Stable v0.2.0"
  - "l2-conv-2d-impl Stable v0.1.0"
key_files:
  created:
    - ".design/main/specifications/l2-conv-2d-impl.md"
  modified:
    - ".design/main/INDEX.md"
    - ".design/main/TASKS.md"
patterns_established:
  - "Filter-major flat []T storage for 2-D: Weights[f*(C_in*K_h*K_w) + c*(K_h*K_w) + i*K_w + j] (CHW, generalises Conv1D Variant A)"
  - "L2 spec Trust Mode promotion: Draft→Stable when MVC satisfied + L1 parent Stable + no RULES conflicts"
duration_minutes: 40
---

# Phase 13 Tasks — Convolutional 2-D Foundation

**Phase:** 13
**Status:** Todo
**Strategic Goal:** Close the 2-D convolution scope deferred in `l1-conv-layers.md` §7. Pre-Planning Stabilization promoted `l1-conv-2d-layers` to Stable v0.2.0 (CHW layout, 9 invariants CONV2D-1..CONV2D-9). This phase finalises the contract side and queues L2 implementation spec authoring; concrete `pkg/layer/conv/` 2-D primitives ship in a follow-up phase scoped after the L2 spec is Stable.

## Atomic Checklist

- [x] [T-13A01] Author L2 implementation spec `l2-conv-2d-impl.md`
- [x] [T-13T01] Validation — verify CONV2D-1..9 invariants are fully mapped in the new L2 spec
- [x] [T-13Z01] Phase 13 gate

## Detailed Tracking

### [T-13A01] Author L2 implementation spec `l2-conv-2d-impl.md`

- **Spec:** [l1-conv-2d-layers.md](../specifications/l1-conv-2d-layers.md) §3 (CONV2D-1..CONV2D-9), §4 (Detailed Design), §6 (Implementation Notes)
- **Status:** Todo
- **Assignment:** Agent (delegated to `/magic-spec` workflow)
- **Verify:**
  - File `.design/main/specifications/l2-conv-2d-impl.md` exists with `Layer: implementation`, `Implements: l1-conv-2d-layers.md`, and Status ∈ `{Draft, RFC, Stable}`.
  - Spec contains §4 `Invariant Compliance` table with one row per CONV2D-1..CONV2D-9 — confirm via: `grep -E "^\| CONV2D-[0-9]" .design/main/specifications/l2-conv-2d-impl.md | wc -l` returns `9`.
  - `.design/main/INDEX.md` registers `l2-conv-2d-impl.md` with matching Status / Version.
  - Pre-flight `node .magic/scripts/executor.js check-prerequisites --json --workspace main` returns `ok: true` with no new `ORPHANED_SPEC` for `l2-conv-2d-impl.md`.
- **Handoff:** Triggers replan via `/magic-task main` — once the L2 spec is Stable, Phase 13 is amended (or a follow-up phase scoped) with implementation tracks for `pkg/layer/conv/conv2d.go`, `maxpool2d.go`, `avgpool2d.go`, `flatten2d.go`, `pkg/nn/options.go` (`WithConv2D` + friends), and the MNIST 2-D adapter (`l2-dataset-loader-impl.md` minor amendment for `WithImageShape`).
- **Notes:** Reuse the structure of `l2-conv-layers-impl.md` (1-D analogue) as a template. CHW layout from CONV2D-C9 is non-negotiable — Invariant Compliance row for CONV2D-9 must call out the exact flat-index formulae verbatim. Weight storage MUST follow filter-major flat `[]T` of length `F * C_in * K_h * K_w` (generalised Conv1D Variant A from `pkg/layer/conv/conv1d.go:26-40`). Document MNIST 2-D adapter requirement in §6 Implementation Notes — the dataset loader currently emits a flat 784-byte tensor and needs a 1×28×28 re-tensoriser before Conv2D can consume it.

### [T-13T01] Validation — verify CONV2D-1..9 invariants are fully mapped

- **Goal:** Confirm the L2 spec authored in T-13A01 substantively addresses every L1 invariant — no placeholder rows in the Invariant Compliance table.
- **Method:**
  - Read the Invariant Compliance table in `l2-conv-2d-impl.md`.
  - For each row CONV2D-1..CONV2D-9, ensure the right-hand cell names at least one concrete implementation element (function name, struct field, package path, or algorithm reference). Empty or stub cells (e.g., "TBD", "—", "see spec") fail validation.
  - Verify §Detailed Design uses CHW row-major layout verbatim from CONV2D-C9 (no HWC drift introduced during authoring).
  - Run `@role:spec-critic` Post-Update Review on the new L2 spec per `spec.md §Post-Update Review` (C24 persona).
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** Manual review of the 9-row table; `@role:spec-critic` review yields zero findings of type `Substantive Compliance` failure.

### [T-13Z01] Phase 13 gate

- **Goal:** Close Phase 13 scoping with both L1 contract Stable and L2 spec authored + registered + validated.
- **Method:**
  - Confirm `l1-conv-2d-layers` is Stable in INDEX.md and file header (already true at scoping time).
  - Confirm `l2-conv-2d-impl.md` exists, is registered in INDEX.md, and has status ∈ `{Draft, RFC, Stable}` — Trust Mode promotion to Stable allowed if MVC + invariant table satisfied.
  - Re-run `node .magic/scripts/executor.js check-prerequisites --json --workspace main` — must return `ok: true` with zero `ORPHANED_SPEC` warnings (excluding the persistent RFC `l2-ai-doc-metadata`).
  - Update phase-13.md frontmatter: `status: Done`, populate `provides` with `["l1-conv-2d-layers Stable v0.2.0", "l2-conv-2d-impl Stable v0.1.0"]`, `key_files.created` with the new L2 spec path, `duration_minutes` with elapsed phase time.
- **Status:** Todo
- **Assignment:** Agent
- **Verify:** `check-prerequisites` clean; phase-13.md frontmatter populated; STATE.md `Next Action` updated to point at the next phase (implementation tracks for Conv2D in `pkg/layer/conv/`).
- **Handoff:** After gate green → re-run `/magic-task main` to scope follow-up implementation phase (proposed: Phase 14 — Conv2D Implementation + MNIST CNN Example, Tracks A `pkg/layer/conv/` + B `pkg/dataset/` adapter + C `examples/mnist_cnn/`).

## Notes

- This phase is a **scoping phase** — typical for the project's pattern when an L1 concept lands but the matching L2 implementation spec hasn't been authored yet. Compare with the original Phase 12 scoping at commit `0168697 chore(main): scope Phase 12 — Meta-Learning Hooks`, which followed the same single-track-A pattern.
- C29 Stdlib-only continues to apply — no third-party dependencies are introduced by Phase 13.
- C30 Coverage floor is **not** enforced at this phase — there is no `pkg/` code yet. The next phase (Conv2D Implementation) will carry the coverage gate.
- Engine drift `.magic/.version` = 2.1.27 vs INDEX snapshot 2.1.25 remains acknowledged (n-branch). Not a blocker for Phase 13.
