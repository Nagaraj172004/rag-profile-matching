"""File-system tools an LLM can call.

Each function returns plain dicts or lists so the result can be sent back to
the model as JSON. Supported resume types are .pdf, .txt, and .docx.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
RESUME_MARKERS = ("skills", "experience", "summary", "education")


def _metadata(path: Path) -> dict:
    """Return name, size, and modified time for a file."""
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _extract_text(path: Path) -> str:
    """Pull plain text out of a PDF, TXT, or DOCX file."""
    extension = path.suffix.lower()
    if extension == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if extension == ".docx":
        from docx import Document

        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Unsupported file type: {0}".format(extension))


def _looks_like_resume(text: str) -> bool:
    """A resume should mention skills, experience, summary, or education."""
    lowered = text.lower()
    return any(marker in lowered for marker in RESUME_MARKERS)


def read_file(filepath: str) -> dict:
    """Read a resume and return its text plus file metadata.

    Args:
        filepath: Path to a .pdf, .txt, or .docx resume.

    Returns:
        A dict with success, content, and metadata. On failure, success is
        false and error explains what went wrong.
    """
    path = Path(filepath)
    if not path.exists():
        return {"success": False, "error": "File not found: {0}".format(filepath)}
    if not path.is_file():
        return {"success": False, "error": "Not a file: {0}".format(filepath)}

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": "Unsupported type '{0}'. Use .pdf, .txt, or .docx.".format(path.suffix),
        }

    try:
        content = _extract_text(path).strip()
    except Exception as exc:
        return {"success": False, "error": "Could not read file: {0}".format(exc)}

    metadata = _metadata(path)
    if not content:
        return {
            "success": False,
            "error": "File has no extractable text.",
            "metadata": metadata,
        }
    if not _looks_like_resume(content):
        return {
            "success": False,
            "error": "Document does not look like a resume. Expected skills, experience, summary, or education.",
            "metadata": metadata,
        }
    return {"success": True, "content": content, "metadata": metadata}


def list_files(directory: str, extension: Optional[str] = None) -> list:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: Folder to scan.
        extension: Optional filter such as ".pdf" or "txt".

    Returns:
        A list of metadata dicts (name, path, size_bytes, modified).
        An empty list means the folder is missing or nothing matched.
    """
    folder = Path(directory)
    if not folder.exists() or not folder.is_dir():
        return []

    wanted = None
    if extension:
        wanted = extension if extension.startswith(".") else ".{0}".format(extension)
        wanted = wanted.lower()

    results = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if wanted and path.suffix.lower() != wanted:
            continue
        results.append(_metadata(path))
    return results


def write_file(filepath: str, content: str) -> dict:
    """Write text to a file, creating parent folders when needed.

    Args:
        filepath: Destination path.
        content: Text to write.

    Returns:
        A dict with success, filepath, and bytes_written, or an error.
    """
    if not isinstance(content, str):
        return {"success": False, "error": "Content must be a string."}

    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return {"success": False, "filepath": str(path), "error": str(exc)}

    return {
        "success": True,
        "filepath": str(path),
        "bytes_written": path.stat().st_size,
    }


def search_in_file(filepath: str, keyword: str) -> dict:
    """Search a file for a keyword and return each hit with nearby lines.

    Matching is case-insensitive. Context is the matching line plus one line
    before and one line after it.

    Args:
        filepath: File to search.
        keyword: Text to find.

    Returns:
        A dict with match_count and a list of matches. On failure, success
        is false and error explains why.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        return {"success": False, "error": "Keyword cannot be empty."}

    loaded = read_file(filepath)
    if not loaded.get("success"):
        return loaded

    lines = loaded["content"].splitlines()
    needle = keyword.lower()
    matches = []
    for index, line in enumerate(lines):
        if needle not in line.lower():
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 2)
        matches.append(
            {
                "line_number": index + 1,
                "line": line,
                "context": "\n".join(lines[start:end]),
            }
        )

    return {
        "success": True,
        "filepath": str(Path(filepath)),
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
    }


TOOLS = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file,
}


def run_tool(name: str, arguments: dict):
    """Call a tool by name. Unknown tools and bad arguments return an error dict."""
    function = TOOLS.get(name)
    if function is None:
        return {"success": False, "error": "Unknown tool: {0}".format(name)}
    try:
        return function(**arguments)
    except TypeError as exc:
        return {"success": False, "error": "Bad arguments for {0}: {1}".format(name, exc)}
