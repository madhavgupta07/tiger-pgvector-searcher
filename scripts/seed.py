"""Generate synthetic readings, embed the notes, and bulk-insert them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pgvector import Vector

from app import db, embeddings, seed_data  # noqa: E402


def main(n: int = 4000) -> None:
    print(f"Generating {n} synthetic readings ...")
    rows = seed_data.generate(n=n)
    print(f"Embedding {len(rows)} notes with {embeddings.get_embedder().name} ...")
    vecs = embeddings.embed_texts([r["note"] for r in rows])
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
    db.close_pool()
    print("Seeded.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
