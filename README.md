# Tiger Vector Search

**AI-powered hybrid vector search with pgvector + pgvectorscale on Tiger Cloud.**

Store high-dimensional embeddings *next to* time-series data in a single PostgreSQL
service, then run similarity searches that filter by **time range**, **metadata**,
and **distance metric** — no separate vector database sidecar.

Built for the MLH **Global Hack Week: Hacktoberfest** challenge
*[Implement AI-Powered Vector Search with pgvector](https://events.mlh.io/events/14553-global-hack-week-hacktoberfest)*.

---

## What it demonstrates

A time-series workload (IoT sensor readings) where every row carries:

- a `timestamptz` time column (hypertable-partitioned),
- relational metadata (`site`, `device_id`, `sensor_type`, `value`, `unit`),
- a natural-language `note`, and
- its **384-dim embedding** (`VECTOR(384)`).

Queries blend semantic similarity with time/metadata filters in one SQL statement.

```
Telemetry (note text)
      │  fastembed / OpenAI
      ▼
  embedding vector(384) ───────────┐
      │                            │
      ▼                            ▼
 readings hypertable          DiskANN index (vectorscale)
  (chunked by day)             BM25 index (pg_textsearch)
      │                            │
      └────────────┬───────────────┘
                   ▼
   hybrid search (RRF) → grounded RAG answer
```

## Stack

| Layer | Choice |
| --- | --- |
| Database | Tiger Cloud (TimescaleDB, PG 18) |
| Vector | `vector` (pgvector 0.8.7) + `vectorscale` 0.9.0 — **StreamingDiskANN** |
| Keyword | `pg_textsearch` 1.5.1 — **BM25** for hybrid search |
| Embeddings | `fastembed` (`BAAI/bge-small-en-v1.5`, ONNX) — pluggable |
| API | FastAPI + psycopg3 (pooled) |
| UI | Single-file HTML dashboard |

## Quickstart

### 1. Provision a Tiger Cloud service

```bash
tiger auth login
tiger service create --name vector-demo   # or reuse an existing service
tiger db uri <service-id> --with-password  # copy this into .env
```

### 2. Configure

```bash
cp .env.example .env
# set DATABASE_URL to the connection string from step 1
pip install -r requirements.txt
pip install fastembed   # for the default local embedding model
```

### 3. Create schema, seed, run

```bash
python scripts/init_db.py          # extensions + hypertable + indexes
python scripts/seed.py 4000        # synthetic readings, embedded
python -m uvicorn app.main:app --port 8077
```

Open <http://127.0.0.1:8077>.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/stats` | Rows, chunks, size, extensions, indexes |
| `POST` | `/api/ingest` | Insert readings (notes are embedded server-side) |
| `POST` | `/api/search` | `mode` = `vector` \| `keyword` \| `hybrid`; filters + `metric` |
| `POST` | `/api/rag` | Retrieve then synthesize a grounded answer |
| `POST` | `/api/seed` | Regenerate the synthetic dataset |

Example:

```bash
curl -s localhost:8077/api/search -H 'content-type: application/json' -d '{
  "query": "vibration that suggests bearing wear",
  "k": 5, "mode": "hybrid", "metric": "cosine",
  "filters": {"sites": ["plant-a"], "sensor_types": ["vibration"]}
}'
```

## How the schema meets the challenge

`app/schema.sql`:

- **Embeddings alongside time-series data** — `readings` is a **hypertable**
  (`create_hypertable`) with a `VECTOR(384)` column in the same row.
- **pgvectorscale** — `USING diskann (embedding vector_cosine_ops)` gives an
  approximate nearest-neighbour index. `EXPLAIN` confirms `Index Scan using
  ..._chunk_readings_embedding_diskann` merged across chunks.
- **pgvector distance metrics** — `<=>` cosine, `<->` L2, `<#>` inner product
  are selectable per request.
- **Time + metadata filtering inside Postgres** — time-range and `site` /
  `device_id` / `sensor_type` predicates are applied in the same plan as the
  vector `ORDER BY ... LIMIT`.
- **Hybrid RAG** — vector results are fused (Reciprocal Rank Fusion) with
  pg_textsearch **BM25** results, then passed to `/api/rag` for a grounded answer.

## Embedding providers

Set `EMBEDDING_PROVIDER` in `.env`:

| Value | Model | Notes |
| --- | --- | --- |
| `fastembed` | `BAAI/bge-small-en-v1.5` | Default. Local ONNX, 384-dim. |
| `local` | any sentence-transformers model | Requires `sentence-transformers`. |
| `openai` | `text-embedding-3-small` | Set `EMBEDDING_DIM=1536`. |
| `hash` | — | Deterministic offline fallback, no downloads. |
| `auto` | — | Tries fastembed → sentence-transformers → hash. |

Keep `EMBEDDING_DIM` consistent with the model — it sizes the `VECTOR(n)` column.

## Layout

```
app/
  config.py       env-driven settings
  db.py           pool + pgvector adapter
  embeddings.py   pluggable embedding providers
  schema.sql      hypertable + DiskANN + BM25 indexes
  seed_data.py    synthetic sensor generator
  rag.py          grounded answer (LLM optional)
  main.py         FastAPI app + search logic
  static/index.html  dashboard
scripts/
  init_db.py      apply schema
  seed.py         embed + bulk insert
screenshots/      submission screenshots
```

## Notes

- `ctid` is not unique across a hypertable's chunks, so hybrid fusion is done in
  Python rather than by joining on `ctid`.
- Only cosine is covered by the DiskANN index in this setup; other metrics fall
  back to an exact scan (`/api/search` reports `indexed: true|false`).
- The LLM synthesis step is optional — without an `OPENAI_API_KEY` the RAG
  endpoint returns a grounded extractive answer from the retrieved rows.
