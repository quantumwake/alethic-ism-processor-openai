# Alethic Instruction-based State Machine (OpenAI Processor)

A microservice processor that handles OpenAI language and visual model requests within the Alethic Instruction-based State Machine (ISM) framework.

## Overview

This project implements:
- OpenAI Chat Completion (GPT) processor
- OpenAI Visual Completion (DALL-E) processor
- Message-driven architecture using NATS message broker
- State persistence using PostgreSQL
- Docker containerization for deployments
- Kubernetes configuration for orchestration

## Features

- Stateful message processing using Alethic ISM Core framework
- Connection to OpenAI API for language and image generation
- Token usage monitoring and tracking
- Stream response support for chat completions
- Usage telemetry support
- Configurable deployment for various environments

## Requirements

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager
- PostgreSQL database
- NATS message broker
- OpenAI API key

## Getting Started

### Setup Development Environment

```shell
# Install UV
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ROUTING_FILE`: Path to the message routing configuration (default: `.routing.yaml`)
- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql://postgres:postgres1@localhost:5432/postgres`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Running Locally

1. Ensure PostgreSQL is running and accessible
2. Configure your `.env` file with required variables
3. Run: `python main.py`

## Docker

### Building the Image

Using Makefile:
```shell
make build IMAGE=yourusername/alethic-ism-processor-openai:tag
```

Or directly:
```shell
docker build -t yourusername/alethic-ism-processor-openai:tag .
```

### Running with Docker

```shell
docker run -d \
  --name alethic-ism-processor-openai \
  -e OPENAI_API_KEY="your_api_key_here" \
  -e LOG_LEVEL=DEBUG \
  -e ROUTING_FILE=/app/repo/.routing.yaml \
  -e DATABASE_URL="postgresql://postgres:postgres1@host.docker.internal:5432/postgres" \
  yourusername/alethic-ism-processor-openai:tag
```

## Kubernetes Deployment

The project includes Kubernetes configuration in the `k8s/` directory:

1. Edit the `k8s/secret.yaml` to include your sensitive information
2. Deploy using:
```shell
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
```

## CI/CD

This project uses GitHub Actions for CI/CD:

- `build-main.yml`: Builds and pushes Docker image on pushes to main branch
- `build-release.yml`: Builds, tags, and deploys when a version tag is pushed

### Versioning and Release

To create a new version release:
```shell
make version
```
This will:
1. Increment the patch version
2. Create a git tag
3. Push the tag to trigger the release workflow

## Dependencies

Key dependencies:
- `alethic-ism-core`: Core ISM framework
- `alethic-ism-db`: Database interface for ISM
- `openai`: OpenAI Python client
- `nats-py`: NATS.io client
- `tiktoken`: OpenAI tokenizer

## License

Alethic ISM is under a DUAL licensing model:

**AGPL v3**  
Intended for academic, research, and nonprofit institutional use. As long as all derivative works are also open-sourced under the same license, you are free to use, modify, and distribute the software.

**Commercial License**
Intended for commercial use, including production deployments and proprietary applications. This license allows for closed-source derivative works and commercial distribution. Please contact us for more information.