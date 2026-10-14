"""FastAPI service: hybrid (vector + keyword + time + metadata) search over
time-series data stored with pgvector / pgvectorscale on Tiger Cloud."""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pgvector import Vector
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from . import config, db, embeddings, rag, seed_data

STATIC_DIR = pathlib.Path(__file__).with_name("static")

# pgvector operators per distance metric.
_DISTANCE_OP = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
# Only cosine is covered by the StreamingDiskANN index; others fall back to a
# sequential scan (still correct, just not accelerated).
_INDEXED_METRICS = {"cosine"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.apply_schema()
    yield
    db.close_pool()


app = FastAPI(title="Tiger Vector Search", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------------------- request models ----------------------------- #

class Reading(BaseModel):
    ts: datetime
    device_id: str
    site: str
    sensor_type: str
    value: float
    unit: str | None = None
    note: str


class IngestRequest(BaseModel):
    readings: list[Reading]


class Filters(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    devices: list[str] | None = None
    sites: list[str] | None = None
    sensor_types: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=100)
    metric: str = "cosine"
    mode: str = "hybrid"  # vector | keyword | hybrid
    filters: Filters = Field(default_factory=Filters)


class RagRequest(BaseModel):
    question: str
    k: int = Field(default=6, ge=1, le=50)
    filters: Filters = Field(default_factory=Filters)


# -------------------------------- helpers -------------------------------- #

def _filter_clause(f: Filters, params: dict) -> str:
    where = []
    if f.start is not None:
        where.append("ts >= %(start)s")
        params["start"] = f.start
    if f.end is not None:
        where.append("ts < %(end)s")
        params["end"] = f.end
    if f.devices:
        where.append("device_id = ANY(%(devices)s)")
        params["devices"] = f.devices
    if f.sites:
        where.append("site = ANY(%(sites)s)")
        params["sites"] = f.sites
    if f.sensor_types:
        where.append("sensor_type = ANY(%(sensor_types)s)")
        params["sensor_types"] = f.sensor_types
    return (" WHERE " + " AND ".join(where)) if where else ""


def _row_to_dict(r: dict) -> dict:
    out = dict(r)
    if isinstance(out.get("ts"), datetime):
        out["ts"] = out["ts"].astimezone(timezone.utc).isoformat()
    if out.get("distance") is not None:
        out["distance"] = float(out["distance"])
    out.pop("embedding", None)
    return out


def _vector_search(req: SearchRequest) -> list[dict]:
    op = _DISTANCE_OP[req.metric]
    params: dict = {"qvec": Vector(embeddings.embed_text(req.query)), "k": req.k}
    where = _filter_clause(req.filters, params)
    sql = f"""
        SELECT ts, device_id, site, sensor_type, value, unit, note,
               embedding {op} %(qvec)s AS distance
        FROM readings{where}
        ORDER BY embedding {op} %(qvec)s
        LIMIT %(k)s
    """
    with db.get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [_row_to_dict(r) for r in cur.fetchall()]


def _keyword_search(req: SearchRequest) -> list[dict]:
    params: dict = {
        "qvec": Vector(embeddings.embed_text(req.query)),
        "query": req.query,
        "k": req.k,
    }
    where = _filter_clause(req.filters, params)
    connector = " AND " if where else " WHERE "
    sql = f"""
        SELECT ts, device_id, site, sensor_type, value, unit, note,
               embedding <=> %(qvec)s AS distance,
               note <@> to_bm25query(%(query)s, 'readings_note_bm25') AS bm25
        FROM readings{where}{connector}note <@> to_bm25query(%(query)s, 'readings_note_bm25') < 0
        ORDER BY bm25
        LIMIT %(k)s
    """
    with db.get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [_row_to_dict(r) for r in cur.fetchall()]


def _row_key(r: dict) -> tuple:
    return (r["ts"], r["device_id"], r["site"], r["sensor_type"], r["value"], r["note"])


def _hybrid_search(req: SearchRequest, pool: int = 40) -> list[dict]:
    """Reciprocal Rank Fusion of the vector and BM25 rankings.

    Fused in Python because a TimescaleDB hypertable is partitioned into chunks,
    so physical ctid values are not unique across the table and cannot be joined
    on. Both candidate lists carry a cosine distance for display.
    """
    vreq = req.model_copy(update={"k": pool, "mode": "vector", "metric": "cosine"})
    kreq = req.model_copy(update={"k": pool, "mode": "keyword"})
    vector_hits = _vector_search(vreq)
    keyword_hits = _keyword_search(kreq)

    scores: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    bm25: dict[tuple, float] = {}

    for rank, r in enumerate(vector_hits, start=1):
        key = _row_key(r)
        rows[key] = r
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    for rank, r in enumerate(keyword_hits, start=1):
        key = _row_key(r)
        rows.setdefault(key, r)
        scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        if r.get("bm25") is not None:
            bm25[key] = r["bm25"]

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: req.k]
    out = []
    for key, score in ranked:
        row = dict(rows[key])
        row["rrf"] = round(score, 6)
        row.pop("bm25", None)
        out.append(row)
    return out


# --------------------------------- routes -------------------------------- #

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_dim": config.EMBEDDING_DIM}


@app.get("/api/stats")
def stats() -> dict:
    with db.get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT extname, extversion FROM pg_extension "
            "WHERE extname IN ('timescaledb','vector','vectorscale','pg_textsearch') ORDER BY extname"
        )
        extensions = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT count(*) AS n FROM readings")
        rows = cur.fetchone()["n"]

        cur.execute(
            "SELECT count(*) AS n FROM timescaledb_information.chunks "
            "WHERE hypertable_name = 'readings'"
        )
        chunks = cur.fetchone()["n"]

        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'readings' ORDER BY indexname"
        )
        indexes = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT pg_size_pretty(hypertable_size('readings')) AS size")
        size = cur.fetchone()["size"]

    return {
        "rows": rows,
        "chunks": chunks,
        "size": size,
        "embedding_dim": config.EMBEDDING_DIM,
        "embedder": embeddings.get_embedder().name,
        "extensions": extensions,
        "indexes": indexes,
    }


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    if not req.readings:
        raise HTTPException(status_code=400, detail="no readings supplied")
    notes = [r.note for r in req.readings]
    vecs = embeddings.embed_texts(notes)
    payload = []
    for r, v in zip(req.readings, vecs):
        d = r.model_dump()
        d["embedding"] = Vector(v)
        payload.append(d)
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO readings (ts, device_id, site, sensor_type, value, unit, note, embedding)
            VALUES (%(ts)s, %(device_id)s, %(site)s, %(sensor_type)s, %(value)s,
                    %(unit)s, %(note)s, %(embedding)s)
            """,
            payload,
        )
    return {"inserted": len(payload)}


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    if req.metric not in _DISTANCE_OP:
        raise HTTPException(status_code=400, detail=f"metric must be one of {list(_DISTANCE_OP)}")
    if req.mode not in {"vector", "keyword", "hybrid"}:
        raise HTTPException(status_code=400, detail="mode must be vector | keyword | hybrid")

    if req.mode == "vector":
        results = _vector_search(req)
    elif req.mode == "keyword":
        results = _keyword_search(req)
    else:
        results = _hybrid_search(req)

    return {
        "query": req.query,
        "mode": req.mode,
        "metric": req.metric,
        "indexed": req.metric in _INDEXED_METRICS,
        "count": len(results),
        "results": results,
    }


@app.post("/api/rag")
def rag_endpoint(req: RagRequest) -> dict:
    sreq = SearchRequest(query=req.question, k=req.k, mode="hybrid", filters=req.filters)
    hits = _hybrid_search(sreq)
    answer, mode = rag.synthesize(req.question, hits)
    return {"question": req.question, "answer": answer, "mode": mode, "context": hits}


@app.post("/api/seed")
def seed(n: int = 4000) -> dict:
    rows = seed_data.generate(n=n)
    notes = [r["note"] for r in rows]
    vecs = embeddings.embed_texts(notes)
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE readings")
        cur.executemany(
            """
            INSERT INTO readings (ts, device_id, site, sensor_type, value, unit, note, embedding)
            VALUES (%(ts)s, %(device_id)s, %(site)s, %(sensor_type)s, %(value)s,
                    %(unit)s, %(note)s, %(embedding)s)
            """,
            [{**r, "embedding": Vector(v)} for r, v in zip(rows, vecs)],
        )
    return {"seeded": len(rows)}


@app.delete("/api/readings")
def clear() -> dict:
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE readings")
    return {"cleared": True}
