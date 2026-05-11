"""Index data/source.md into the chunks table.

Run:
    python -m scripts.index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.db import sync_connect
from app.embeddings import embed_batch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "source.md"


def load_source(path: Path) -> str:
    if not path.exists():
        sys.exit(f"Source file not found: {path}")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Split by tokens. tiktoken's cl100k_base is OpenAI-style — close enough
    for sizing purposes; we don't need exact match with sentence-transformers
    tokenization, only consistent chunk granularity."""
    enc = tiktoken.get_encoding("cl100k_base")

    def length_in_tokens(s: str) -> int:
        return len(enc.encode(s))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=length_in_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source_name: str,
    truncate: bool,
) -> None:
    settings = get_settings()
    assert all(len(v) == settings.embedding_dim for v in embeddings), "embedding dim mismatch"

    with sync_connect() as conn:
        with conn.cursor() as cur:
            if truncate:
                cur.execute("TRUNCATE TABLE chunks")
            for i, (text, vec) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{source_name}::{i:05d}"
                cur.execute(
                    """
                    INSERT INTO chunks (id, content, source, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content   = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (chunk_id, text, source_name, i, vec),
                )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to .md or .pdf")
    parser.add_argument(
        "--truncate",
        action="store_true",
        default=True,
        help="Wipe the chunks table before insert (default: true)",
    )
    parser.add_argument("--no-truncate", dest="truncate", action="store_false")
    args = parser.parse_args()

    settings = get_settings()
    source_path = Path(args.source)
    source_name = source_path.stem

    print(f"Loading {source_path}")
    text = load_source(source_path)
    print(f"  {len(text):,} characters")

    print(f"Chunking (size={settings.chunk_size_tokens}, overlap={settings.chunk_overlap_tokens})")
    chunks = chunk_text(
        text,
        chunk_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    print(f"  {len(chunks)} chunks")

    print(f"Embedding with {settings.embedding_model}")
    embeddings = embed_batch(chunks)

    print(f"Upserting into chunks table (truncate={args.truncate})")
    upsert_chunks(chunks, embeddings, source_name=source_name, truncate=args.truncate)

    print("Done.")


if __name__ == "__main__":
    main()
