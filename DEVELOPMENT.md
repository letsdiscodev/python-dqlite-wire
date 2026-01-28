# Development Guide

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.13
uv pip install -e ".[dev]"
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
.venv/bin/pytest

# Run with verbose output
.venv/bin/pytest -v

# Run with coverage
.venv/bin/pytest --cov=dqlitewire
```

## Linting

```bash
# Check for issues
.venv/bin/ruff check src tests

# Auto-fix issues
.venv/bin/ruff check --fix src tests
```

## Formatting

```bash
# Format all files
.venv/bin/ruff format src tests

# Check formatting without modifying
.venv/bin/ruff format --check src tests
```

## Type Checking

```bash
# Run mypy with strict mode
.venv/bin/mypy src
```

## Pre-commit Workflow

Before committing, run all checks:

```bash
.venv/bin/ruff format src tests
.venv/bin/ruff check --fix src tests
.venv/bin/mypy src
.venv/bin/pytest
```
