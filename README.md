# rag-application

This project contains a web application enabling the user to chat with an LLM-based chatbot with access to additional context information, leveraging the RAG paradigm.

## Development

It is advisable to use the VSCode's Development Container mechanism for development. The default configuration allows to set up a Docker container with all dependencies and functionalities, such as Docker-in-Docker.

### Setup

After opening the Devcontainer, it is recommended to use only the setup commands specified in the `justfile`:

```bash
just setup-dev
```

### Running checks

In order to run unit tests, integration tests and other checks just use one of the pre-made commands, e.g.:

```bash
just run-pre-commit
```

The commands are configured to be reused by GitHub Actions jobs.

## Usage

Prerequisities for using the project outside of the Development Container:
1. `just` - for running the setup scripts defined in the `justfile`
2. `docker-compose` - for running multi-container application

In order to set up the whole system, run:

```bash
just up-services
```

To stop it:

```bash
just down-services
```

## Changelog

Refer to `Changelog.md`.
