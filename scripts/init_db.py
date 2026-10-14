"""Create extensions, the hypertable, and indexes on the Tiger Cloud service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, db  # noqa: E402


def main() -> None:
    print(f"Applying schema (embedding dim = {config.EMBEDDING_DIM}) ...")
    db.apply_schema()
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT extname, extversion FROM pg_extension "
            "WHERE extname IN ('timescaledb','vector','vectorscale','pg_textsearch') ORDER BY extname"
        )
        for name, ver in cur.fetchall():
            print(f"  extension  {name:<14} {ver}")
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='readings' ORDER BY 1"
        )
        for (name,) in cur.fetchall():
            print(f"  index      {name}")
    db.close_pool()
    print("Schema ready.")


if __name__ == "__main__":
    main()
