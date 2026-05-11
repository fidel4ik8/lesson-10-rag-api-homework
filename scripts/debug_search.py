"""Quick diagnostic: shows what embed_text returns and how pgvector ranks chunks.

Run:
    python -m scripts.debug_search "your question"
"""

from __future__ import annotations

import sys

from app.db import sync_connect
from app.embeddings import embed_text


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "How should logs be handled?"

    print("=" * 70)
    print(f"QUERY: {q!r}")
    print("=" * 70)

    # 1) What embed_text returns
    print("\n--- STEP 1: embed_text(query) ---")
    qv = embed_text(q)
    print(f"type:           {type(qv).__name__}")
    print(f"len:            {len(qv)}    (= embedding_dim)")
    print(f"first 5 values: {qv[:5]}")
    print(f"last 5 values:  {qv[-5:]}")
    print(f"min:            {min(qv):.6f}")
    print(f"max:            {max(qv):.6f}")
    norm = sum(x * x for x in qv) ** 0.5
    print(f"L2 norm:        {norm:.6f}    (~1.0 because normalize_embeddings=True)")

    # 2) Top-3 search via pgvector
    print("\n--- STEP 2: pgvector top-3 ---")
    sql = """
        SELECT
            id,
            embedding <=> %s::vector       AS distance,
            1 - (embedding <=> %s::vector) AS similarity,
            LEFT(content, 100)             AS preview
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 3
    """
    from psycopg import ClientCursor

    with sync_connect() as conn:
        # Show the SQL template (what we wrote)
        print("SQL template:")
        print(sql)

        # ClientCursor.mogrify() renders the EXACT string that would be sent
        # to Postgres, with parameters substituted client-side. Useful for
        # debugging only — production uses server-side prepared statements.
        with ClientCursor(conn) as ccur:
            rendered = ccur.mogrify(sql, (qv, qv, qv))
        if len(rendered) > 400:
            rendered = (
                rendered[:200]
                + f" ...[truncated {len(rendered) - 400} chars]... "
                + rendered[-200:]
            )
        print("SQL actually sent to Postgres (vectors truncated):")
        print(rendered)
        print()

        cur = conn.cursor()

        # Now execute and fetch
        cur.execute(sql, (qv, qv, qv))
        header = "{:<16} {:>10} {:>12}   {}".format("id", "distance", "similarity", "preview")
        print(header)
        print("-" * 100)
        for cid, dist, sim, prev in cur.fetchall():
            print(f"{cid:<16} {dist:>10.4f} {sim:>12.4f}   {prev}")

        # Bonus: EXPLAIN — shows whether HNSW index is actually used
        print("\n--- STEP 3: EXPLAIN (query plan) ---")
        cur.execute("EXPLAIN " + sql, (qv, qv, qv))
        for (line,) in cur.fetchall():
            print(line)


if __name__ == "__main__":
    main()
