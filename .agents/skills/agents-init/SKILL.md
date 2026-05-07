---
name: agents-init
description: >
  Universal agent environment initializer. Creates junctions/symlinks and
  hardlinks so any supported AI agent can discover workflows, skills, and
  rules from the canonical .agents/ directory.
argument-hint: "[claude|gemini|qwen|copilot|codex|kilocode|all] (default: claude)"
---

# Agents Init Skill

Sets up the mapping between the canonical `.agents/` directory and
agent-specific config directories/files. Works for **any project** — just
place `.agents/` at the repo root and run the skill.

## Usage

```
/agents-init              # default: claude  (→ .claude/ + CLAUDE.md + GEMINI.md)
/agents-init qwen         # only qwen        (→ .qwen/  + QWEN.md)
/agents-init claude qwen  # both agents
/agents-init all          # every supported agent
```

## Supported Agents

| Token | Config dir | Instruction files |
| --- | --- | --- |
| `claude` | `.claude/` | `CLAUDE.md` |
| `gemini` | `.gemini/` | `GEMINI.md` |
| `qwen` | `.qwen/` | `QWEN.md` |
| `cursor` | `.cursor/rules/` | *(rules linked directly)* |
| `copilot` | `.github/` | `COPILOT.md` |
| `codex` | `.codex/` | `CODEX.md` |
| `kilocode` | `.kilocode/` | *(rules linked directly)* |
| `lingma` | `.lingma/` | *(rules linked directly)* |

> **Default** (no argument) = `claude`.

## Canonical Source

`.agents/` is always the single source of truth:

```
.agents/
  workflows/   ← agent "commands"
  skills/      ← agent skill libraries
  rules/       ← shared rule files
```

## Bootstrap (First Run)

The skill cannot be invoked as `/agents-init` until `.claude/skills/` exists.
On the very first setup, run the script directly from the project root:

**Windows:**

```powershell
powershell -NoProfile -File .agents/skills/agents-init/scripts/setup_windows.ps1
# or with arguments:
powershell -NoProfile -File .agents/skills/agents-init/scripts/setup_windows.ps1 -Agents qwen
powershell -NoProfile -File .agents/skills/agents-init/scripts/setup_windows.ps1 -Agents all
```

**Unix/macOS:**

```bash
bash .agents/skills/agents-init/scripts/setup_unix.sh
# or with arguments:
bash .agents/skills/agents-init/scripts/setup_unix.sh qwen
bash .agents/skills/agents-init/scripts/setup_unix.sh all
```

After the first run `.claude/skills/` will be a junction/symlink to
`.agents/skills/`, and the skill becomes available as `/agents-init` for
subsequent re-initialization.

## What Each Run Does

1. Parses target agents from arguments (default: `claude`).
2. For each target agent:
   - Creates the config directory if absent.
   - Removes stale junctions / symlinks.
   - Creates fresh junctions (Windows) or symlinks (Unix):
     - `<dir>/workflows` → `.agents/workflows`
     - `<dir>/skills`    → `.agents/skills`
     - `<dir>/rules`     → `.agents/rules`
   - Creates hardlinks for agent instruction files (`CLAUDE.md`, `QWEN.md`, …)
     pointing at `AGENTS.md`.
3. Removes linked paths from the git index (`git rm --cached`).
4. Prints a verification summary.

## Agent Registry

Agent configuration is defined in [`agents.json`](agents.json).
To add a new agent — edit only that file; the scripts read it automatically
(once wired up for JSON parsing).

```jsonc
{
    "claude": {
        "dir": ".claude",              // config directory at repo root
        "commands": "commands",        // target name for workflows junction/symlink (null = skip)
        "skills": "skills",            // target name for skills junction/symlink
        "rules": null,                 // target name for rules junction/symlink
        "files": ["CLAUDE.md"],        // hardlinks → AGENTS.md
        "description": "Claude Code (Anthropic)"
    },
    ...
}
```

## Resources

- [agents.json](agents.json)                             — agent registry
- [scripts/setup_windows.ps1](scripts/setup_windows.ps1) — PowerShell (Windows)
- [scripts/setup_unix.sh](scripts/setup_unix.sh)         — Bash (Unix/macOS)

## Windows Junction Safety

When managing Windows junctions (`mklink /J`) and git index, follow this strict order to prevent data loss:

### The Problem

`git rm -r --cached <path>` on Windows **follows junctions** and physically deletes files in the junction target, even with `--cached`. Example: `git rm -r --cached .claude/commands` where `.claude/commands` is a junction to `workflows/` will **delete all files in `workflows/` from disk**.

### Safe Procedure

Always run `git rm --cached` **before** creating junctions, while the paths are empty or nonexistent:

1. `git rm --cached`   ← first, while no junctions exist yet
2. `mklink /J ...`     ← then create junctions

When removing from git index, list **specific file paths** rather than directories:

```bash
# Safe — specific files only
git rm --cached --ignore-unmatch workflows/magic.analyze.md

# Dangerous — git will traverse the junction into parent/source directories
git rm -r --cached .claude/commands
```
