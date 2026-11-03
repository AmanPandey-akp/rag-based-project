"""Read notes from disk: .md, .txt and .pdf (page-aware)."""

from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_file(path: Path) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). Page is None for non-PDF files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
        return pages
    raise ValueError(f"Unsupported file type: {path.suffix}")
