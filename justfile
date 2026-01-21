# --------------------------------------------------
# This file contains setup scripts for the project.
# For more info, see: https://github.com/casey/just
# --------------------------------------------------

set shell := ["bash", "-c"]

up-infrastructure:
    docker compose up --build