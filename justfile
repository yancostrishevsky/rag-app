# --------------------------------------------------
# This file contains setup scripts for the project.
# For more info, see: https://github.com/casey/just
# --------------------------------------------------

set shell := ["bash", "-c"]

# Run static repository checks
run-pre-commit:
    uv run pre-commit run --all-files

# Set up environment and install dependencies
setup-dev:
    uv venv
    uv sync --project . --extra dev
