"""Chunk resumes by section, embed them, and store them in ChromaDB.

Resumes are loaded with list_files and read_file from fs_tools.py. Each
section (Summary, Skills, Experience, Education) becomes its own chunk so a
job can match the relevant part of a resume. Embeddings come from the
Hugging Face model all-MiniLM-L6-v2.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fs_tools import list_files, read_file

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "resumes"
SECTION_NAMES = ("summary", "skills", "experience", "education", "projects")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_model = None
_client = None


def _allow_model_download() -> None:
    """This network replaces HTTPS certificates, so the default check fails."""
    import requests
    import urllib3
    from huggingface_hub import configure_http_backend

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def backend_factory():
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)


def get_model():
    """Load the embedding model once per process."""
    global _model
    if _model is None:
        _allow_model_download()
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_client(persist_dir: str = "chroma_db"):
    """Open the on-disk Chroma client once per process."""
    global _client
    if _client is None:
        import chromadb

        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def get_collection(persist_dir: str = "chroma_db"):
    """Return the resume collection. Run build_index first."""
    return get_client(persist_dir).get_collection(COLLECTION_NAME)


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pack(section: str, text: str) -> list:
    """Keep short sections whole. Split long ones with a small overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [{"section": section, "text": text}]

    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        piece = text[start:end].strip()
        if piece:
            parts.append({"section": section, "text": piece})
        if end >= len(text):
            break
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start
    return parts


def chunk_by_section(text: str) -> list:
    """Split a resume on section headings such as Skills and Experience."""
    chunks = []
    current = "header"
    buffer = []
    for line in _clean(text).splitlines():
        key = line.strip().lower()
        if key in SECTION_NAMES:
            chunks.extend(_pack(current, "\n".join(buffer)))
            current = key
            buffer = [line]
        else:
            buffer.append(line)
    chunks.extend(_pack(current, "\n".join(buffer)))
    return chunks


def _section_body(text: str, section: str) -> str:
    collecting = False
    body = []
    for line in _clean(text).splitlines():
        key = line.strip().lower()
        if key in SECTION_NAMES:
            if collecting:
                break
            collecting = key == section
            continue
        if collecting:
            body.append(line)
    return "\n".join(body).strip()


def extract_metadata(text: str, resume_path: str) -> dict:
    """Pull name, skills, years of experience, and education from a resume."""
    lines = [line.strip() for line in _clean(text).splitlines() if line.strip()]
    name = lines[0] if lines else "Unknown"
    years_match = re.search(r"(\d+)\s*\+?\s*years", _clean(text), flags=re.IGNORECASE)
    years = int(years_match.group(1)) if years_match else 0
    return {
        "candidate_name": name,
        "skills": " ".join(_section_body(text, "skills").split()),
        "experience_years": years,
        "education": " ".join(_section_body(text, "education").split()),
        "resume_path": resume_path,
    }


def relative_resume_path(filepath: str) -> str:
    """Store resume paths with forward slashes, relative to the project root."""
    path = Path(filepath)
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative = path
    return relative.as_posix()


def iter_resumes(persist_dir: str = "chroma_db") -> dict:
    """Group stored chunks by resume path."""
    collection = get_collection(persist_dir)
    payload = collection.get(include=["metadatas", "documents"])
    grouped = {}
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    for document, meta in zip(documents, metadatas):
        slot = grouped.setdefault(meta["resume_path"], {"meta": meta, "chunks": []})
        slot["chunks"].append({"section": meta["section"], "text": document})
    return grouped


def build_index(resumes_dir: str = "resumes", persist_dir: str = "chroma_db") -> dict:
    """Read every resume, embed its chunks, and replace the Chroma collection."""
    started = time.perf_counter()
    files = list_files(resumes_dir)
    ids = []
    documents = []
    metadatas = []
    resume_count = 0

    for item in files:
        loaded = read_file(item["path"])
        if not loaded.get("success"):
            print("skip {0}: {1}".format(item["path"], loaded.get("error")))
            continue
        resume_count += 1
        resume_path = relative_resume_path(item["path"])
        meta = extract_metadata(loaded["content"], resume_path)
        for index, chunk in enumerate(chunk_by_section(loaded["content"])):
            ids.append("{0}::{1}::{2}".format(Path(resume_path).name, chunk["section"], index))
            documents.append(chunk["text"])
            chunk_meta = dict(meta)
            chunk_meta["section"] = chunk["section"]
            metadatas.append(chunk_meta)

    if not documents:
        raise SystemExit("No resumes indexed. Run python generate_resumes.py first.")

    vectors = get_model().encode(documents, show_progress_bar=False)
    client = get_client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors.tolist(),
        metadatas=metadatas,
    )
    return {
        "resumes": resume_count,
        "chunks": len(ids),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    stats = build_index()
    print(
        "Indexed {resumes} resumes into {chunks} chunks in {seconds} seconds.".format(**stats)
    )


if __name__ == "__main__":
    main()
