-- Tiger Vector Search schema
-- Requires a Tiger Cloud (TimescaleDB) service with pgvector + pgvectorscale.
-- The {dim} placeholder is substituted at apply time from the embedding config.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- Time-series readings, each row carrying a natural-language note and its
-- embedding. Time, metadata and vector live in the SAME row / table.
CREATE TABLE IF NOT EXISTS readings (
    ts          TIMESTAMPTZ       NOT NULL,
    device_id   TEXT              NOT NULL,
    site        TEXT              NOT NULL,
    sensor_type TEXT              NOT NULL,
    value       DOUBLE PRECISION  NOT NULL,
    unit        TEXT,
    note        TEXT              NOT NULL,
    embedding   VECTOR({dim})     NOT NULL
);

-- Turn it into a hypertable partitioned by time (one chunk per day).
SELECT create_hypertable(
    'readings',
    'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Relational / time indexes used by the metadata + time-range filters.
CREATE INDEX IF NOT EXISTS readings_device_ts_idx
    ON readings (device_id, ts DESC);
CREATE INDEX IF NOT EXISTS readings_site_sensor_ts_idx
    ON readings (site, sensor_type, ts DESC);

-- pgvectorscale StreamingDiskANN index for approximate nearest-neighbour search.
CREATE INDEX IF NOT EXISTS readings_embedding_diskann
    ON readings USING diskann (embedding vector_cosine_ops)
    WITH (num_neighbors = 50);

-- pg_textsearch BM25 index for the keyword half of hybrid search.
CREATE INDEX IF NOT EXISTS readings_note_bm25
    ON readings USING bm25 (note) WITH (text_config = 'english');
