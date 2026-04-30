"""
pdf_loader.py
Handles PDF/text file parsing and splitting into chunks for embedding.
"""

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file (Streamlit UploadedFile)."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return full_text.strip()


def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from a plain .txt file."""
    return uploaded_file.read().decode("utf-8")


def load_document(uploaded_file) -> str:
    """Route to the correct extractor based on file type."""
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Split raw text into overlapping chunks for better retrieval.
    Smaller chunks = more precise retrieval.
    Overlap = avoids losing context at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 30]
