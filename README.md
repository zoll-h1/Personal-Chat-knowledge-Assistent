# RAG Chat Assistant

A local-first, privacy-focused RAG system that answers questions using your exported ChatGPT conversations as a private knowledge base.

## Why This Project
Chat history contains durable context you already paid to create: design decisions, debugging steps, code snippets, and architecture rationale. This assistant turns those conversations into a searchable, citation-grounded knowledge system without requiring a paid API.

## Features
- Ingest ChatGPT export ZIP/JSON/HTML with defensive parsing
- Privacy-first redaction before processed storage and embeddings
- Chat-aware chunking (message-level, code blocks preserved, overlap)
- Local embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + Qdrant
- Filtered retrieval by topic/date/chat IDs with citations
- Strict anti-hallucination abstain path (`Insufficient context`)
- FastAPI REST API + Streamlit web UI
- Docker Compose stack (`qdrant`, `api`, `ui`)
- Minimal evaluation suite (Hit@5, latency, abstain rate, cost=0)

## Architecture
```text
                    +------------------------------+
                    | ChatGPT Export ZIP/JSON/HTML |
                    +---------------+--------------+
                                    |
                                    v
+------------------+     +-------------------------+
| Redaction Engine | --> | Normalized messages.jsonl |
+------------------+     +-------------------------+
                                    |
                                    v
                           +------------------+
                           | Chat Chunker      |
                           | chunks.jsonl      |
                           +---------+--------+
                                     |
                                     v
                         +-------------------------+
                         | Local Embeddings (ST)   |
                         +------------+------------+
                                      |
                                      v
                           +---------------------+
                           | Qdrant (COSINE)     |
                           +----------+----------+
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
+---------------------------+                    +---------------------------+
| FastAPI /ask, /admin/*    |                    | Streamlit UI              |
| retrieval + answer modes  |                    | query + filters + upload |
+---------------------------+                    +---------------------------+
```

## Quickstart (3 commands)
```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api python scripts/smoke_test.py
```

Then open:
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

## After You Copy Your ZIP To `data/raw`
If your export file is `data/raw/my_chatgpt_export.zip`, run:

```bash
docker compose exec api python scripts/ingest_export.py --input data/raw/my_chatgpt_export.zip
docker compose exec api python scripts/smoke_test.py
```

You can also ingest via API:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"input_path":"data/raw/my_chatgpt_export.zip"}'
```

## Ingestion Flow
1. Read export ZIP/JSON/HTML from `data/raw`.
2. Parse JSON/HTML variants and branching message trees.
3. Normalize to `data/processed/messages.jsonl` schema:
   - `chat_id`, `chat_title`, `message_id`, `parent_message_id`, `role`, `created_at`, `text`, `has_code`, `attachments`, `topic`, `source`
4. Redact PII/secrets before disk write and before embeddings.
5. Infer lightweight topic labels.
6. Build chunk records in `data/processed/chunks.jsonl`.
7. Embed + index into Qdrant collection.

## Privacy / Redaction
Redacted patterns:
- email -> `[REDACTED_EMAIL]`
- phone -> `[REDACTED_PHONE]`
- tokens/keys (OpenAI, HF, AWS, JWT-like) -> `[REDACTED_TOKEN]`
- password patterns -> `[REDACTED_PASSWORD]`

Controls:
- `ALLOWLIST_IT_ONLY=true` to keep only IT topics
- `EXCLUDE_TITLE_KEYWORDS=a,b,c` to drop chats by title keywords

## API
### `GET /health`
Liveness check.

### `POST /ask`
Request:
```json
{
  "question": "How did I run FastAPI in my old notes?",
  "top_k": 10,
  "topic": "fastapi",
  "date_from": "2024-01-01T00:00:00Z",
  "date_to": "2024-12-31T23:59:59Z",
  "chat_ids": ["chat-sample-1"],
  "mode": "extractive"
}
```

Response:
```json
{
  "answer": "Run: uvicorn app.api.main:app --reload [C1]",
  "citations": [
    {
      "chat_id": "chat-sample-1",
      "message_ids": ["m-2"],
      "snippet": "Run: `uvicorn app.api.main:app --reload` ...",
      "score": 0.82,
      "created_at": "2024-01-10T09:01:00Z"
    }
  ],
  "confidence": 0.82,
  "latency_ms": 33.4
}
```

If confidence is too low, the assistant abstains:
- Starts answer with `Insufficient context`
- Shows closest snippets and asks for a narrower query

### Admin endpoints
- `POST /admin/ingest`
- `POST /admin/reindex`
- `GET /admin/stats`
- `POST /admin/collection/reset`
- `DELETE /admin/chats/{chat_id}`

## Streamlit UI
- Upload export file (or ingest from path)
- Ask questions with filters (topic/date/top_k/chat_ids)
- Select mode: `extractive` (default) or `llm` (optional)
- View answer, confidence, latency, and expandable citations

## Evaluation
Run:
```bash
python -m app.eval.run_eval --dataset app/eval/dataset.example.json --top-k 5
```

Sample metrics table:

| Metric | Value |
|---|---:|
| Hit@5 | 0.67 |
| Avg latency (ms) | 41.2 |
| Abstain rate | 0.33 |
| Embedding cost | 0 (local model) |

## Demo Script
`scripts/smoke_test.py` demonstrates:
1. Collection reset
2. Ingest `data/raw/sample_export_stub.json`
3. Run `/ask`
4. Print answer + top citation

Run:
```bash
docker compose exec api python scripts/smoke_test.py
```

## Local Dev Commands
```bash
# API (without docker)
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# UI
streamlit run app/ui/streamlit_app.py --server.port 8501

# Ingest export
python scripts/ingest_export.py --input data/raw/sample_export_stub.json

# Reindex
python scripts/reindex.py --reset

# Tests
pytest -q
```

## Deployment Notes
- Designed for local Docker runtime.
- Uses local embeddings and local Qdrant.
- Optional `MODE=llm` requires `OPENAI_API_KEY`; retrieval remains local.

## Future Improvements
- Better HTML export coverage and attachment OCR pipeline
- Cross-encoder reranker option for higher precision
- Multi-tenant auth and encrypted-at-rest payloads
- Incremental ingestion and deduplication of repeated turns
- Richer eval set with answer faithfulness scoring
