import os
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFIngestionError(Exception):
    """Base exception for PDF ingestion errors."""
    pass


class UnsupportedPDFError(PDFIngestionError):
    """Raised when a PDF cannot be processed (e.g. scanned/image-only with no extractable text)."""
    pass


def load_and_split_pdf(
    file_path: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Loads a PDF file, validates text content & structure, normalizes page metadata,
    and splits the document into text chunks using RecursiveCharacterTextSplitter.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Maximum character count per chunk (default: 1000).
        chunk_overlap: Character overlap between consecutive chunks (default: 150).

    Returns:
        List of chunked LangChain Document objects with normalized metadata.

    Raises:
        PDFIngestionError: If file is missing, invalid extension, corrupted, or zero pages.
        UnsupportedPDFError: If PDF contains no extractable text (e.g. scanned/image-only).
    """
    path_obj = Path(file_path)

    # 1. Validate file existence
    if not path_obj.exists() or not path_obj.is_file():
        raise PDFIngestionError(f"PDF file not found at path: {file_path}")

    # 2. Validate PDF file extension
    if path_obj.suffix.lower() != ".pdf":
        raise PDFIngestionError(f"Invalid file type '{path_obj.suffix}'. Expected a '.pdf' file.")

    # 3. Load PDF pages using PyPDFLoader
    try:
        loader = PyPDFLoader(str(path_obj))
        raw_docs = loader.load()
    except Exception as e:
        raise PDFIngestionError(f"Failed to parse PDF file '{path_obj.name}': {e}") from e

    # 4. Check for zero pages
    if not raw_docs:
        raise PDFIngestionError(f"PDF file '{path_obj.name}' contains zero pages.")

    # 5. Validate extractable text across pages (detect scanned/image-only PDFs)
    total_text = "".join(doc.page_content for doc in raw_docs).strip()
    if not total_text:
        raise UnsupportedPDFError(
            f"PDF file '{path_obj.name}' contains no extractable text. Scanned or image-only PDFs are not supported."
        )

    # 6. Normalize page metadata (1-indexed page integer & filename source)
    filename = path_obj.name
    normalized_docs = []
    for idx, doc in enumerate(raw_docs):
        # Only keep text-bearing pages or all pages, but normalize metadata
        page_num = idx + 1
        doc.metadata["source"] = filename
        doc.metadata["page"] = page_num
        normalized_docs.append(doc)

    # 7. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(normalized_docs)

    # Re-verify metadata on chunks
    for chunk in chunks:
        chunk.metadata["source"] = filename
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1

    return chunks
