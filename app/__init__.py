"""Tiger Vector Search — AI-powered hybrid vector search on Tiger Cloud.

Stores high-dimensional embeddings alongside time-series data in a single
PostgreSQL/TimescaleDB service using pgvector + pgvectorscale (DiskANN) and
pg_textsearch (BM25), then serves similarity search and RAG over the results.
"""

__version__ = "1.0.0"
