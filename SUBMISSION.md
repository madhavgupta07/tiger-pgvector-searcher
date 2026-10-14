# Submission — Implement AI-Powered Vector Search with pgvector

- **Event:** Global Hack Week: Hacktoberfest (`019f70e5-2d9b-76f8-9a2c-fc9661592609`)
- **Challenge:** Implement AI-Powered Vector Search with pgvector (`01a11b66-3a25-902e-3690-f9ea40cd5014`)
- **What to submit:** "A screenshot of and/or a link to your vector database."

## What was built

A FastAPI service on **Tiger Cloud (TimescaleDB)** that stores 384-dim
embeddings in a **hypertable** alongside time-series sensor data, indexed with
pgvectorscale **StreamingDiskANN**, and searched with vector distance + time +
metadata filters in one SQL query. Hybrid search fuses the vector ranking with
**pg_textsearch BM25** ranking, and a `/api/rag` endpoint returns a grounded
answer over the retrieved rows.

## Evidence

### Vector database (from the running app)

```
rows                4,000
hypertable chunks   8
table size          13 MB
vector dim          384
embedder            fastembed:BAAI/bge-small-en-v1.5
extensions          pg_textsearch 1.5.1 · timescaledb 2.30.2 · vector 0.8.7 · vectorscale 0.9.0
```

Indexes on `readings`:

```
readings_embedding_diskann     USING diskann (embedding vector_cosine_ops)
readings_note_bm25             USING bm25 (note)
readings_device_ts_idx         (device_id, ts DESC)
readings_site_sensor_ts_idx    (site, sensor_type, ts DESC)
readings_ts_idx                (ts DESC)
```

### Index plan proof

`EXPLAIN (ANALYZE)` for a filtered vector query merges the DiskANN index across
time chunks and applies the metadata filter in the same plan:

```
Merge Append
  Sort Key: (embedding <=> '<query vector>'::vector)
  ->  Index Scan using _hyper_1_9_chunk_readings_embedding_diskann on _hyper_1_9_chunk
        Order By: (embedding <=> '<query vector>'::vector)
        Filter: (site = 'plant-a'::text)
  ->  Index Scan using _hyper_1_10_chunk_readings_embedding_diskann on _hyper_1_10_chunk
        ...
```

### Screenshots (`screenshots/`)

| File | Shows |
| --- | --- |
| `01-vector-search.png` | Hybrid search results + database panel |
| `02-rag-answer.png` | Grounded RAG answer over retrieved rows |
| `03-vector-database.png` | DB stats + index definitions |

## Reproduce

```bash
cp .env.example .env            # set DATABASE_URL from: tiger db uri <id> --with-password
pip install -r requirements.txt
python scripts/init_db.py
python scripts/seed.py 4000
python -m uvicorn app.main:app --port 8077
```

## To submit (your call)

1. Push this repo to a public GitHub repo and/or use the screenshots above.
2. On the GHW submissions page (or via the DevRelay gateway `enter_challenge`),
   enter the project for challenge `01a11b66-3a25-902e-3690-f9ea40cd5014` and
   attach the screenshot / repo link.
