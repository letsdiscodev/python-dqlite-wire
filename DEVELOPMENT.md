# Development Guide

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --group dev
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

## Pre-commit Workflow

Before committing, run all checks:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
uv run mypy src
uv run pytest
```
