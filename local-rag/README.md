# Local Multimodal RAG — Updated Pipeline

This version changes the ingestion order to:

```text
PDF
  ↓
Extraction + provenance
  ↓
Normalization
  ↓
Generic section / hierarchy detection
  ↓
OCR + Vision enrichment
  ↓
Section-aware chunking
  ↓
Embeddings
  ↓
PostgreSQL + pgvector
```

The important design rule is that **document structure is determined from original extracted text/layout**, while OCR and Vision outputs are stored as secondary evidence. Generated visual descriptions therefore do not create section boundaries.

## Models

The Docker containers call the Ollama server running on the Mac host:

- `nomic-embed-text` for embeddings (768 dimensions)
- `qwen3:4b` for answer generation
- `qwen3-vl:4b` for visual interpretation
- `deepseek-ocr` for OCR

Docker uses `host.docker.internal`, not `localhost`, to reach Ollama on the host.

## Quick start

Copy this codebase into your existing `local-rag` project, keeping your existing `documents/` directory and PDFs.

```bash
docker compose build

docker compose up -d rag-postgres rag-api
```

For fast development without the expensive OCR/Vision pass:

```bash
docker compose run --rm rag-ingestion python -m app.pipeline --force --skip-vision
```

For the full multimodal pass:

```bash
docker compose run --rm rag-ingestion python -m app.pipeline --force
```

The full pass can be slow on a Mac because every extracted visual may trigger OCR and Vision model calls.

## Inspect generated artifacts

For each document the ingestion worker writes:

```text
extraction/<document>/
├── document.json
├── document_normalized.json
├── document_structure.json
├── document_enriched.json
└── chunks.json
```

`document_structure.json` is the key checkpoint for section detection. `document_enriched.json` adds OCR/Vision evidence without replacing source text.

## Database

`document_chunks` now stores:

- `chunk_type`
- `section_path`
- `element_ids`
- `evidence`
- `chunk_hash`
- `embedding`

The application also performs idempotent schema upgrades at runtime, so the existing database does not require a destructive reset.

## API

```bash
curl 'http://localhost:8000/health'
curl 'http://localhost:8000/ask?question=What%20is%20ImageNet-C%3F'
```

## Reference implementation consulted

The structure follows patterns seen in public multimodal RAG projects that retain text, tables and images as separate evidence and preserve document/page provenance. One example is `multimodal-rag-team/multimodal-rag-platform`, which uses typed PDF blocks and PostgreSQL/pgvector, and `jndumu/multi-modal-rag`, which uses structured document categories, OCR and structure-aware chunking. We did not copy their code; the implementation here remains local-model/Ollama based and keeps the current project simple. citehttps://github.com/multimodal-rag-team/multimodal-rag-platformturn814735search0
