import os
from typing import List, Dict
from pypdf import PdfReader
from docx import Document

def parse_pdf(file_path: str) -> List[Dict]:
    """Parse PDF using pypdf (much faster than unstructured)."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n\n"
    return [{"text": text, "metadata": {"source": os.path.basename(file_path), "type": "pdf"}}]

def parse_docx(file_path: str) -> List[Dict]:
    """Parse DOCX using python-docx."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return [{"text": text, "metadata": {"source": os.path.basename(file_path), "type": "docx"}}]

def parse_txt_md(file_path: str) -> List[Dict]:
    """Parse TXT and Markdown files."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    file_type = "md" if file_path.endswith(".md") else "txt"
    return [{"text": text, "metadata": {"source": os.path.basename(file_path), "type": file_type}}]

def ingest_document(file_path: str) -> List[Dict]:
    """Route file to correct parser based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in [".txt", ".md"]:
        return parse_txt_md(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
