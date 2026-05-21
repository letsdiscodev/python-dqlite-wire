#!/usr/bin/env bash
# check-commit-msg.sh — guard against workflow vocabulary in commit messages.
#
# The project convention forbids workflow tokens ("Round N", "Cycle N",
# "Phase N", "Bundle X", "ultrathink", "ISSUE-N", references to .md
# filenames) in commit messages. Three published commits (19ae33e8,
# d806944d, 4b516dee) leaked "round 29"/"round 30" through review;
# this script is the forward-looking guard so future commits do not
# regress. Published history is left intact (amending would rewrite
# SHAs other packages reference).
#
# Usage:
#   # Lint a single message file (e.g. as a commit-msg hook):
#   scripts/check-commit-msg.sh .git/COMMIT_EDITMSG
#
#   # Lint the most recent commit body:
#   git log -1 --format=%B | scripts/check-commit-msg.sh -
#
#   # Lint a range of commits (e.g. everything since main diverged):
#   scripts/check-commit-msg.sh --range origin/main..HEAD
#
# Install as a git hook:
#   ln -s ../../scripts/check-commit-msg.sh .git/hooks/commit-msg
#
# Exit codes:
#   0 — clean
#   1 — workflow vocabulary detected (with file:line context)
#   2 — usage error
#
# Regex notes (per reviewer): a naive \b(round|cycle|phase|bundle)\s+\d+\b
# also flags legitimate phrasing like "phase 1 of two" or "cycle 3 of
# the test". We use the capital-leading form so casual prose is not
# falsely accused; the leaked commits all used the capital-leading
# pattern ("round 29", "round 30") when spoken as a workflow token.
# Also flags ISSUE-N tokens and bundle references that survived prior
# scrubs.

set -euo pipefail

# Capital-leading workflow tokens. Matches "Round 29", "round 29",
# "Cycle 3", etc. Case-insensitive so the lowercase-leading variants
# the leaked commits used are caught, but the tighter wording
# (\s+\d+\b) avoids matching "phaseout 2024" or "round-trip 3".
WORKFLOW_RE='\b(round|cycle|phase|bundle)\s+[0-9]+\b'

# ISSUE-N tokens (e.g. ISSUE-T4, ISSUE-42) and done/ filename refs.
ISSUE_RE='\bISSUE-[A-Z0-9]+\b'
DONE_RE='\bdone/[A-Za-z0-9._-]+\.md\b'

# "ultrathink" / "this round" / "next round" prose markers.
PROSE_RE='\b(ultrathink|this round|next round|prior round|earlier round)\b'

scan_text() {
    # $1 — label (file path or commit SHA) for the report.
    # stdin — message body to scan.
    local label="$1"
    local body
    body="$(cat)"
    local rc=0
    # Strip comment lines (git commit message convention: '#' at column 0).
    local stripped
    stripped="$(printf '%s\n' "$body" | grep -v '^#' || true)"

    if printf '%s\n' "$stripped" | grep -inE "$WORKFLOW_RE" >/dev/null; then
        printf 'check-commit-msg: %s: workflow token (Round/Cycle/Phase/Bundle N) detected\n' "$label" >&2
        printf '%s\n' "$stripped" | grep -inE "$WORKFLOW_RE" >&2 || true
        rc=1
    fi
    if printf '%s\n' "$stripped" | grep -inE "$ISSUE_RE" >/dev/null; then
        printf 'check-commit-msg: %s: ISSUE-N token detected\n' "$label" >&2
        printf '%s\n' "$stripped" | grep -inE "$ISSUE_RE" >&2 || true
        rc=1
    fi
    if printf '%s\n' "$stripped" | grep -inE "$DONE_RE" >/dev/null; then
        printf 'check-commit-msg: %s: done/ filename reference detected\n' "$label" >&2
        printf '%s\n' "$stripped" | grep -inE "$DONE_RE" >&2 || true
        rc=1
    fi
    if printf '%s\n' "$stripped" | grep -inE "$PROSE_RE" >/dev/null; then
        printf 'check-commit-msg: %s: workflow prose marker detected\n' "$label" >&2
        printf '%s\n' "$stripped" | grep -inE "$PROSE_RE" >&2 || true
        rc=1
    fi
    return $rc
}

usage() {
    cat >&2 <<'EOF'
Usage:
  check-commit-msg.sh <file>           lint a single commit-message file
  check-commit-msg.sh -                lint message read from stdin
  check-commit-msg.sh --range <rev>    lint every commit in the rev range
EOF
    exit 2
}

if [ $# -lt 1 ]; then
    usage
fi

case "$1" in
    --range)
        if [ $# -ne 2 ]; then
            usage
        fi
        rc=0
        for sha in $(git rev-list "$2"); do
            if ! git log -1 --format=%B "$sha" | scan_text "$sha"; then
                rc=1
            fi
        done
        exit $rc
        ;;
    -)
        scan_text "<stdin>"
        ;;
    -h|--help)
        usage
        ;;
    *)
        if [ ! -f "$1" ]; then
            printf 'check-commit-msg: %s: not a file\n' "$1" >&2
            exit 2
        fi
        scan_text "$1" < "$1"
        ;;
esac
