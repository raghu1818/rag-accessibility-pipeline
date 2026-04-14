# AccessBot RAG Pipeline

> **Production-grade Retrieval-Augmented Generation for the Accessibility Research Lab**
>
> Helping blind and low-vision users navigate product manuals through natural-language
> queries — with zero hallucination guarantees and chain-of-thought transparency.

[![CI](https://github.com/your-org/rag-accessibility-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rag-accessibility-pipeline/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/your-org/rag-accessibility-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/rag-accessibility-pipeline)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Why This Exists](#why-this-exists)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Multi-Agent Design](#multi-agent-design)
7. [Zero-Hallucination Strategy](#zero-hallucination-strategy)
8. [PDF Ingestion & OCR Fallback](#pdf-ingestion--ocr-fallback)
9. [Configuration](#configuration)
10. [Running Tests](#running-tests)
11. [Docker & Docker Compose](#docker--docker-compose)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Contributing](#contributing)

---

## Why This Exists

Screen-reader users face a specific problem: product manuals are PDF-heavy, poorly
structured, and inaccessible when a user cannot see the page layout.  Asking a chatbot
a question like "how do I reset the hearing aid?" and receiving a hallucinated answer
is not just unhelpful — it can be genuinely dangerous.

AccessBot solves this by:

- Indexing uploaded manuals into a FAISS vector store with semantic chunking.
- Retrieving the most relevant passages before generation (RAG pattern).
- Enforcing a **two-stage hallucination guard**: every factual claim must be traceable
  to a retrieved passage, or the system refuses to answer.
- Writing answers in plain language, with numbered steps and inline source citations,
  optimised for screen-reader consumption.

---

## Architecture

```
                              USER / SCREEN READER
                                      │
                              ┌───────▼────────┐
                              │   FastAPI REST  │
                              │     API         │
                              │  (port 8000)    │
                              └──┬──────────┬───┘
                                 │          │
                    POST /ingest │          │ POST /query
                                 │          │
                    ┌────────────▼──┐   ┌───▼────────────────┐
                    │   Ingestion   │   │    LangGraph        │
                    │    Agent      │   │    Pipeline         │
                    │               │   │                     │
                    │ ┌───────────┐ │   │  ┌──────────────┐  │
                    │ │PDF        │ │   │  │  Retrieval   │  │
                    │ │Processor  │ │   │  │  Agent       │  │
                    │ │           │ │   │  │              │  │
                    │ │ Native ───┤ │   │  │ FAISS top-k  │  │
                    │ │ extract   │ │   │  │ similarity   │  │
                    │ │           │ │   │  │ search       │  │
                    │ │ OCR  ─────┤ │   │  └──────┬───────┘  │
                    │ │ fallback  │ │   │         │          │
                    │ └───────────┘ │   │  ┌──────▼───────┐  │
                    └───────┬───────┘   │  │  Generation  │  │
                            │           │  │  Agent       │  │
                            │           │  │              │  │
                    ┌───────▼───────┐   │  │ CoT prompt   │  │
                    │  FAISS Vector │◄──┘  │ + citations  │  │
                    │  Store        │      └──────┬───────┘  │
                    │               │             │          │
                    │  Chunked docs │      ┌──────▼───────┐  │
                    │  + metadata   │      │ Hallucination│  │
                    │  Embeddings   │      │ Guard Node   │  │
                    │  (OpenAI)     │      │              │  │
                    └───────────────┘      │ Verify every │  │
                                           │ claim ───────┼──┘
                                           │ retry once   │
                                           └──────────────┘
                                                  │
                                           FINAL ANSWER
                                        (grounded + cited)
```

### Technology Stack

| Layer            | Technology                         |
|------------------|------------------------------------|
| Orchestration    | LangGraph 0.2 (state machine)      |
| LLM              | OpenAI GPT-4o (configurable)       |
| Embeddings       | OpenAI text-embedding-3-small      |
| Vector store     | FAISS (CPU, persistent)            |
| PDF extraction   | PyMuPDF (native) + Tesseract (OCR) |
| API framework    | FastAPI + Uvicorn                  |
| Config           | Pydantic Settings v2               |
| Logging          | structlog (JSON in prod)           |
| Containers       | Docker multi-stage + Compose       |
| CI/CD            | GitHub Actions                     |

---

## Project Structure

```
rag-accessibility-pipeline/
│
├── src/
│   ├── agents/
│   │   ├── ingestion_agent.py   # PDF → chunks → FAISS upsert
│   │   ├── retrieval_agent.py   # FAISS similarity search
│   │   └── generation_agent.py  # LLM generation + hallucination guard
│   │
│   ├── api/
│   │   ├── main.py              # FastAPI app factory + lifespan
│   │   ├── routes/
│   │   │   ├── health.py        # GET /health, /health/ready
│   │   │   ├── ingest.py        # POST /ingest (multipart PDF)
│   │   │   └── query.py         # POST /query
│   │   └── schemas/
│   │       ├── ingest.py        # Pydantic I/O models
│   │       └── query.py
│   │
│   ├── core/
│   │   ├── config.py            # Pydantic Settings (env vars)
│   │   ├── logging.py           # structlog setup
│   │   └── exceptions.py        # Domain exception hierarchy
│   │
│   ├── graph/
│   │   ├── pipeline.py          # LangGraph graph wiring
│   │   └── state.py             # Shared PipelineState schema
│   │
│   ├── processors/
│   │   ├── pdf_processor.py     # PyMuPDF extraction + chunking
│   │   └── ocr_processor.py     # Tesseract wrapper + image pre-proc
│   │
│   ├── prompts/
│   │   └── templates.py         # All prompt strings (CoT + guard)
│   │
│   └── vector_store/
│       └── faiss_store.py       # Thread-safe FAISS wrapper
│
├── tests/
│   ├── conftest.py              # Shared fixtures, mock LLM/store
│   ├── test_processors.py       # PDF + OCR unit tests
│   ├── test_agents.py           # Agent unit tests
│   └── test_api.py              # FastAPI integration tests
│
├── docker/
│   ├── Dockerfile               # Multi-stage production image
│   └── Dockerfile.dev           # Dev image with hot-reload
│
├── .github/workflows/
│   ├── ci.yml                   # Lint → test → docker build
│   └── cd.yml                   # Tag → build → push to GHCR
│
├── docker-compose.yml           # Production deployment
├── docker-compose.dev.yml       # Local development
├── pyproject.toml               # Project metadata + tool config
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev/test dependencies
└── .env.example                 # Template for secrets
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH`
- An OpenAI API key

### 1. Clone and set up

```bash
git clone https://github.com/your-org/rag-accessibility-pipeline.git
cd rag-accessibility-pipeline

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Ingest a manual

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/product_manual.pdf"
```

### 5. Query it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I enable accessibility mode?"}'
```

---

## API Reference

### `GET /health`
Liveness probe — returns 200 if the process is running.

```json
{
  "status": "ok",
  "environment": "production",
  "document_count": 142
}
```

### `GET /health/ready`
Readiness probe — used by Kubernetes / load balancers.

---

### `POST /ingest`
Upload a PDF for processing and indexing.

**Request:** `multipart/form-data`, field `file` must be a `.pdf`.

**Response (202):**
```json
{
  "message": "Successfully indexed 'hearing_aid_manual.pdf'.",
  "source": "/app/data/raw/hearing_aid_manual_a1b2c3d4.pdf",
  "filename": "hearing_aid_manual.pdf",
  "chunk_count": 87,
  "pages": 24,
  "extraction_methods": ["native", "ocr"],
  "error": null
}
```

**Errors:**
- `415` — not a PDF
- `413` — file exceeds MAX_UPLOAD_SIZE_MB
- `422` — text extraction failed

---

### `POST /query`
Ask a question about the indexed documents.

**Request body:**
```json
{
  "question": "How do I change the battery?",
  "score_threshold": 0.3,
  "include_thinking": false
}
```

**Response (200):**
```json
{
  "answer": "To change the battery, open the battery door on the bottom of the device by sliding it to the left. [Source 1]  Insert a size-13 battery with the positive (+) side facing up. [Source 1]",
  "grounded": true,
  "sources": [
    {"filename": "hearing_aid_manual.pdf", "page": 8, "extraction_method": "native"}
  ],
  "thinking": null,
  "error": null
}
```

Set `include_thinking: true` to receive the chain-of-thought reasoning block.

---

## Multi-Agent Design

The pipeline uses **LangGraph** to wire three specialised agents into a directed graph:

```
RETRIEVAL ──► GENERATION ──► HALLUCINATION_GUARD
                  ▲                  │
                  └──── retry once ──┘ (if grounded == false)
```

### Ingestion Agent (`src/agents/ingestion_agent.py`)
- Receives a file path from the API.
- Calls `PDFProcessor` to extract and chunk text.
- De-duplicates by deleting stale chunks for the same source before re-indexing.
- Upserts Document objects into `FAISSVectorStore`.

### Retrieval Agent (`src/agents/retrieval_agent.py`)
- Reads `state.query` from the LangGraph state.
- Runs `FAISSVectorStore.similarity_search()` with a configurable score threshold.
- Populates `state.retrieved_documents`.

### Generation Agent (`src/agents/generation_agent.py`)
- Formats retrieved documents into a numbered `CONTEXT` block.
- Calls GPT-4o with the chain-of-thought + grounding prompt.
- Parses the `<thinking>…</thinking>` block from the response.
- On retry (when `state.grounded == False`), appends a stricter correction hint.

### Hallucination Guard (inside Generation Agent)
- Sends the draft answer + context to a second LLM call.
- Parses the JSON `{"grounded": bool, "violations": [...]}` response.
- If violations are found, routes back to the generation node for one retry.

---

## Zero-Hallucination Strategy

AccessBot enforces grounding at three levels:

| Level | Mechanism |
|-------|-----------|
| **Prompt** | System prompt explicitly forbids stating facts not in the context; requires `[Source N]` inline citations |
| **Chain-of-thought** | `<thinking>` block forces the model to identify gaps before writing the final answer |
| **Guard node** | A separate LLM call verifies every factual sentence against the context passages; violations trigger a retry |

If the second attempt still contains violations, the guard marks `grounded: false` in the
response so callers can surface a warning to the user.

---

## PDF Ingestion & OCR Fallback

```
PDF file
   │
   ├─► PyMuPDF page.get_text()
   │       │
   │       ├── chars ≥ 50? ──► native text  ──► chunk ──► FAISS
   │       │
   │       └── chars < 50? ──► render page as 300 DPI PNG
   │                                │
   │                        pytesseract.image_to_string()
   │                                │
   │                           ┌────▼─────┐
   │                           │ pre-proc │
   │                           │ greyscale│
   │                           │ upscale  │
   │                           │ sharpen  │
   │                           │ contrast │
   │                           └────┬─────┘
   │                                │
   │                           OCR text ──► chunk ──► FAISS
   │
   └─► Metadata: source, page, chunk, extraction_method
```

Each Document chunk carries full provenance metadata so AccessBot can tell users
exactly which page and file a piece of information came from.

---

## Configuration

All configuration is via environment variables (or a `.env` file).  See `.env.example`
for the full list.  Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `FAISS_TOP_K` | `5` | Number of passages to retrieve |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |
| `OCR_DPI` | `300` | Rendering DPI for OCR pages |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `ENVIRONMENT` | `production` | `development` enables coloured logs |

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_api.py -v
```

The test suite uses mocked LLM and vector store calls so no OpenAI key is needed.
A real Tesseract installation is required for `tests/test_processors.py`.

---

## Docker & Docker Compose

### Production

```bash
# Build
docker compose build

# Run (requires OPENAI_API_KEY in .env)
docker compose up -d

# View logs
docker compose logs -f api

# Tear down
docker compose down
```

### Development (hot-reload)

```bash
docker compose -f docker-compose.dev.yml up
```

The development image mounts the project directory so code changes are reflected
without rebuilding.

### Image details

The production image uses a **two-stage build**:
1. **Builder** — installs Python packages with all build tools present.
2. **Runtime** — copies only the installed packages; no compiler toolchain in the
   final image.  Runs as a non-root user (`appuser:appgroup`).

---

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Triggered on every push to `main`/`develop` and all pull requests.

```
┌─────────────────────────────────────────────────────────────┐
│  CI Pipeline                                                 │
│                                                              │
│  lint ──► test (py3.11, py3.12) ──► docker-build            │
│                                                              │
│  lint:                                                       │
│    • ruff check (lint)                                       │
│    • ruff format --check                                     │
│    • mypy --strict                                           │
│                                                              │
│  test:                                                       │
│    • Install Tesseract on runner                             │
│    • pytest --cov=src --cov-fail-under=70                    │
│    • Upload coverage to Codecov                              │
│                                                              │
│  docker-build:                                               │
│    • Build production image                                  │
│    • Smoke-test with python -c import                        │
└─────────────────────────────────────────────────────────────┘
```

### Continuous Deployment (`.github/workflows/cd.yml`)

Triggered on semver tags (`v*.*.*`).

1. Logs into GitHub Container Registry (GHCR).
2. Builds and pushes the image with `latest`, `v1.2`, and `v1.2.3` tags.
3. Creates a GitHub Release with auto-generated release notes.

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes and add tests.
3. Run `ruff check . && ruff format . && pytest` — all must pass.
4. Open a pull request against `main`.

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) format
for commit messages.

---

## License

MIT © Accessibility Research Lab
