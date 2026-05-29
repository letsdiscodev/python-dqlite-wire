# Development Guide

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra dev
```

## Development Tools

This project uses modern Python tooling:

| Tool | Purpose | Command |
|------|---------|---------|
| **pytest** | Testing framework | `pytest` |
| **ruff** | Linter (replaces flake8, isort, etc.) | `ruff check` |
| **ruff format** | Code formatter (replaces black) | `ruff format` |
| **mypy** | Static type checker | `mypy src` |

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=dqlitewire
```

## Linting

```bash
# Check for issues
uv run ruff check src tests

# Auto-fix issues
uv run ruff check --fix src tests
```

## Formatting

```bash
# Format all files
uv run ruff format src tests

# Check formatting without modifying
uv run ruff format --check src tests
```

## Type Checking

```bash
# Run mypy with strict mode
uv run mypy src
```

## Comments and docstrings

Keep comments and docstrings to a minimum — the code should be clear
enough to stand on its own. Prefer renaming a function or variable over
writing a comment to explain an unclear one.

- Write a comment only when it captures something genuinely non-obvious
  that the code cannot: a subtle invariant, a security caveat, a
  workaround for an upstream bug, or a "why" a reader would otherwise
  get wrong. Delete comments that merely restate what the code does.
- Most functions need no docstring. Add a one-line docstring only when
  it tells a reader something the name and signature do not. Avoid
  multi-paragraph essays, param-by-param prose, and "Notes/Divergence"
  sections.
- Do not record rationale, history, or decision logs in comments — that
  context belongs in the commit message and git history (`git blame`,
  `git log -p`), where it stays attached to the change instead of aging
  in the source.
- Tooling directives (`# type:`, `# noqa`, `# pragma:`) are exempt —
  keep them.

When in doubt, leave it out: a missing explanation is a `git blame`
away; a redundant or stale comment is noise every future reader pays
for.

## Pre-commit Workflow

Before committing, run all checks:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
uv run mypy src
uv run pytest
```

## Commit Message Conventions

Commit messages MUST NOT contain workflow vocabulary. The forbidden
tokens are:

- Capital-leading "Round N", "Cycle N", "Phase N", "Bundle N" (and the
  lowercase variants) when used as workflow markers.
- `ISSUE-N` issue identifiers (e.g. `ISSUE-T4`, `ISSUE-1438`).
- References to `done/<name>.md` filenames.
- Prose markers like "ultrathink", "this round", "next round",
  "prior round", "earlier round".

The substantive citation in a commit body should always be expressed
as the durable name it points at (constant name, contract name, public
symbol) — never as the workflow step that introduced it.

A forward-looking guard lives at `scripts/check-commit-msg.sh`. Run
it against the most recent commit:

```bash
git log -1 --format=%B | scripts/check-commit-msg.sh -
```

Or against a range of commits:

```bash
scripts/check-commit-msg.sh --range origin/main..HEAD
```

To install it as a local commit-msg hook:

```bash
ln -s ../../scripts/check-commit-msg.sh .git/hooks/commit-msg
```

Three legacy commits (`19ae33e8`, `d806944d`, `4b516dee`) carry
"round 29" / "round 30" markers in their bodies. These are published
and have not been amended (rewriting SHAs would break cross-package
commit references); the linter is the regression-prevention measure.
