"""Parser for Markdown knowledge base documents with YAML front-matter."""
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import yaml

from src.kb.schemas import DocumentMetadata, DocChunk


def parse_frontmatter(content: str, filename: str) -> Tuple[DocumentMetadata, str]:
    """
    Extract YAML front-matter and document body.
    """
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    
    if not match:
        # Fallback if no front-matter
        meta = DocumentMetadata(
            document_id=filename.replace(".md", "").upper(),
            title=filename,
            status="active",
            filename=filename
        )
        return meta, content

    yaml_text, body = match.groups()
    data: Dict[str, Any] = yaml.safe_load(yaml_text) or {}
    
    # Normalize string dates and booleans
    meta = DocumentMetadata(
        document_id=str(data.get("document_id", filename.replace(".md", ""))),
        title=str(data.get("title", filename)),
        status=str(data.get("status", "active")).lower(),
        effective_date=str(data.get("effective_date", "")) if data.get("effective_date") else None,
        last_reviewed=str(data.get("last_reviewed", "")) if data.get("last_reviewed") else None,
        audience=str(data.get("audience", "customer")).lower(),
        policy_authority=str(data.get("policy_authority", "official")).lower(),
        supersedes=str(data.get("supersedes", "")) if data.get("supersedes") else None,
        superseded_by=str(data.get("superseded_by", "")) if data.get("superseded_by") else None,
        customer_answering=bool(data.get("customer_answering", True)),
        filename=filename,
    )
    return meta, body


def chunk_document(file_path: Path) -> List[DocChunk]:
    """
    Split a Markdown document into semantic chunks by H2 sections,
    preserving metadata, document title, and section headings.
    """
    filename = file_path.name
    content = file_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(content, filename)

    # Extract H1 title if present
    h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    doc_title = h1_match.group(1).strip() if h1_match else meta.title

    chunks: List[DocChunk] = []

    # Split on H2 headers (## Heading)
    # Regex splits while keeping delimiters or we can iterate lines
    lines = body.splitlines()
    current_heading = "Overview"
    current_lines: List[str] = []
    chunk_index = 0

    for line in lines:
        if line.startswith("## "):
            # Save previous section if it has non-trivial text
            section_content = "\n".join(current_lines).strip()
            # Ignore H1 title line in overview if that's all it was
            clean_content = re.sub(r"^#\s+.*$", "", section_content, flags=re.MULTILINE).strip()
            if clean_content:
                citation = f"{filename} > {current_heading}"
                full_text = (
                    f"Document: {meta.title} ({filename})\n"
                    f"Status: {meta.status} | Authority: {meta.policy_authority}\n"
                    f"Section: {current_heading}\n\n"
                    f"{clean_content}"
                )
                chunks.append(
                    DocChunk(
                        chunk_id=f"{meta.document_id}_{chunk_index}",
                        document_id=meta.document_id,
                        filename=filename,
                        title=doc_title,
                        heading=current_heading,
                        content=clean_content,
                        full_text=full_text,
                        metadata=meta,
                        citation=citation,
                    )
                )
                chunk_index += 1
            current_heading = line.replace("## ", "").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Append the last section
    section_content = "\n".join(current_lines).strip()
    clean_content = re.sub(r"^#\s+.*$", "", section_content, flags=re.MULTILINE).strip()
    if clean_content:
        citation = f"{filename} > {current_heading}"
        full_text = (
            f"Document: {meta.title} ({filename})\n"
            f"Status: {meta.status} | Authority: {meta.policy_authority}\n"
            f"Section: {current_heading}\n\n"
            f"{clean_content}"
        )
        chunks.append(
            DocChunk(
                chunk_id=f"{meta.document_id}_{chunk_index}",
                document_id=meta.document_id,
                filename=filename,
                title=doc_title,
                heading=current_heading,
                content=clean_content,
                full_text=full_text,
                metadata=meta,
                citation=citation,
            )
        )

    return chunks


def load_all_chunks(kb_dir: Path) -> List[DocChunk]:
    """Load and chunk all Markdown files in the knowledge base directory."""
    all_chunks: List[DocChunk] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        chunks = chunk_document(md_file)
        all_chunks.extend(chunks)
    return all_chunks
