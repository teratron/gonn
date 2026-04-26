#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS INIT (UNIX)
# Universal agent environment initializer.
#
# Usage:
#   setup_unix.sh              # default: claude
#   setup_unix.sh qwen
#   setup_unix.sh claude qwen
#   setup_unix.sh all
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY_FILE="$(dirname "$SCRIPT_DIR")/agents.json"

# ───────────────────────────────────────────────────────────────────────────────
# 1. Agent Registry — loaded from agents.json
# ───────────────────────────────────────────────────────────────────────────────

if [ ! -f "$REGISTRY_FILE" ]; then
    echo "ERROR: Registry not found: $REGISTRY_FILE" >&2
    exit 1
fi

# Parse agents.json with python3 (preferred) or jq as fallback
_json_get() {
    local agent="$1" field="$2"
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
val = data.get(sys.argv[2], {}).get(sys.argv[3])
if val is None: print('')
elif isinstance(val, list): print(' '.join(val))
else: print(val)
" "$REGISTRY_FILE" "$agent" "$field"
    elif command -v jq &>/dev/null; then
        if [ "$field" = "files" ]; then
            jq -r ".\"$agent\".\"$field\" | if type == \"array\" then join(\" \") else \"\" end" "$REGISTRY_FILE"
        else
            jq -r ".\"$agent\".\"$field\" | if . == null then \"\" else . end" "$REGISTRY_FILE"
        fi
    else
        echo "ERROR: python3 or jq required to parse agents.json" >&2
        exit 1
    fi
}

# Build in-memory lookup arrays
declare -A AGENT_DIR
declare -A AGENT_FILES
declare -A AGENT_WORKFLOWS
declare -A AGENT_SKILLS
declare -A AGENT_RULES

ALL_AGENTS=$(
    if command -v python3 &>/dev/null; then
        python3 -c "import json, sys; print(' '.join(json.load(open(sys.argv[1])).keys()))" "$REGISTRY_FILE"
    else
        jq -r 'keys[]' "$REGISTRY_FILE" | tr '\n' ' '
    fi
)

for agent in $ALL_AGENTS; do
    AGENT_DIR[$agent]="$(_json_get "$agent" dir)"
    AGENT_FILES[$agent]="$(_json_get "$agent" files)"
    AGENT_WORKFLOWS[$agent]="$(_json_get "$agent" workflows)"
    AGENT_SKILLS[$agent]="$(_json_get "$agent" skills)"
    AGENT_RULES[$agent]="$(_json_get "$agent" rules)"
done

# ───────────────────────────────────────────────────────────────────────────────
# 2. Parse arguments
# ───────────────────────────────────────────────────────────────────────────────

if [ $# -eq 0 ]; then
    TARGETS="claude"
elif [ "$1" = "all" ]; then
    TARGETS="$ALL_AGENTS"
else
    TARGETS="$*"
fi

# Validate
for t in $TARGETS; do
    if [ -z "${AGENT_DIR[$t]+x}" ]; then
        echo "ERROR: Unknown agent '$t'. Supported: $ALL_AGENTS" >&2
        exit 1
    fi
done

echo ">>> Initializing Unix Agent Environment for: $TARGETS"

# ───────────────────────────────────────────────────────────────────────────────
# 3. Helpers
# ───────────────────────────────────────────────────────────────────────────────

remove_existing() {
    local path="$1"
    if [ -L "$path" ] || [ -e "$path" ]; then
        echo "  Removing: $path"
        rm -rf "$path"
    fi
}

make_symlink() {
    local link="$1"
    local target="$2"
    remove_existing "$link"
    ln -s "$target" "$link"
    echo "  Symlink: $link → $target"
}

make_hardlink() {
    local link="$1"
    local target="$2"
    remove_existing "$link"
    ln "$target" "$link"
    echo "  Hardlink: $link → $target"
}

# ───────────────────────────────────────────────────────────────────────────────
# 4. Git index cleanup (before creating links)
# ───────────────────────────────────────────────────────────────────────────────

echo ""
echo "Synchronizing git index (pre-link)..."

git_paths=()
for agent in $TARGETS; do
    dir="${AGENT_DIR[$agent]}"
    [ -n "${AGENT_WORKFLOWS[$agent]}" ] && git_paths+=("$dir/${AGENT_WORKFLOWS[$agent]}")
    [ -n "${AGENT_SKILLS[$agent]}" ] && git_paths+=("$dir/${AGENT_SKILLS[$agent]}")
    [ -n "${AGENT_RULES[$agent]}" ] && git_paths+=("$dir/${AGENT_RULES[$agent]}")
    for f in ${AGENT_FILES[$agent]}; do
        git_paths+=("$f")
    done
done

if [ ${#git_paths[@]} -gt 0 ]; then
    # shellcheck disable=SC2068
    git rm -r --cached --ignore-unmatch "${git_paths[@]}" 2>/dev/null || true
fi

# ───────────────────────────────────────────────────────────────────────────────
# 5. Create symlinks and hardlinks per agent
# ───────────────────────────────────────────────────────────────────────────────

for agent in $TARGETS; do
    dir="${AGENT_DIR[$agent]}"
    files="${AGENT_FILES[$agent]}"

    echo ""
    echo "[+] Agent: $agent  →  $dir"

    mkdir -p "$dir"

    # Calculate relative path depth for symlink target
    # .claude/commands → ../../.agents/workflows  (depth = number of slashes in dir + 1)
    depth=$(echo "$dir" | tr -cd '/' | wc -c)
    prefix=""
    for ((i=0; i<=depth; i++)); do prefix="../$prefix"; done

    [ -n "${AGENT_WORKFLOWS[$agent]}" ] && make_symlink "$dir/${AGENT_WORKFLOWS[$agent]}"   "${prefix}.agents/workflows"
    [ -n "${AGENT_SKILLS[$agent]}" ]    && make_symlink "$dir/${AGENT_SKILLS[$agent]}" "${prefix}.agents/skills"
    [ -n "${AGENT_RULES[$agent]}" ]  && make_symlink "$dir/${AGENT_RULES[$agent]}"  "${prefix}.agents/rules"

    for f in $files; do
        make_hardlink "$f" "AGENTS.md"
    done
done

# ───────────────────────────────────────────────────────────────────────────────
# 6. Verification
# ───────────────────────────────────────────────────────────────────────────────

echo ""
echo ">>> Verification:"
for agent in $TARGETS; do
    dir="${AGENT_DIR[$agent]}"
    paths=""
    [ -n "${AGENT_WORKFLOWS[$agent]}" ] && paths="$paths $dir/${AGENT_WORKFLOWS[$agent]}"
    [ -n "${AGENT_SKILLS[$agent]}" ]    && paths="$paths $dir/${AGENT_SKILLS[$agent]}"
    [ -n "${AGENT_RULES[$agent]}" ]  && paths="$paths $dir/${AGENT_RULES[$agent]}"
    [ -n "$paths" ] && ls -ld $paths 2>/dev/null || true
    for f in ${AGENT_FILES[$agent]}; do
        ls -li "$f" 2>/dev/null || true
    done
done

echo ""
echo ">>> Done."
