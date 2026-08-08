import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.ingestion import load_and_split_pdf

load_dotenv()

# Environment configuration with fallback defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "vectorstore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kt_rag_collection")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))


def get_embeddings(model_name: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """Initializes GoogleGenerativeAIEmbeddings using specified or configured model."""
    model = model_name or EMBEDDING_MODEL
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing or empty.")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None
) -> Chroma:
    """Returns a persistent Chroma vector store instance."""
    p_dir = persist_directory or PERSIST_DIRECTORY
    c_name = collection_name or COLLECTION_NAME
    embeddings = embedding_function if embedding_function is not None else get_embeddings()

    return Chroma(
        collection_name=c_name,
        embedding_function=embeddings,
        persist_directory=p_dir
    )


def generate_deterministic_id(doc: Document, idx: int) -> str:
    """Generates a deterministic unique ID for a document chunk to prevent duplicate indexing."""
    source = doc.metadata.get("source", "doc")
    page = doc.metadata.get("page", 1)
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}_p{page}_idx{idx}_{content_hash}"


def index_documents(
    documents: List[Document],
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None
) -> List[str]:
    """
    Indexes a list of Document objects into the persistent Chroma vector store with deterministic IDs
    to prevent duplicate chunks upon re-indexing.

    Args:
        documents: List of chunked Document objects with metadata.
        persist_directory: Custom persistence directory path.
        collection_name: Custom Chroma collection name.
        embedding_function: Custom embedding instance (e.g. for offline unit testing).

    Returns:
        List of generated document IDs in Chroma.
    """
    if not documents:
        return []

    vector_store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    # Generate deterministic IDs for duplicate prevention
    doc_ids = [generate_deterministic_id(doc, idx) for idx, doc in enumerate(documents)]
    
    return vector_store.add_documents(documents, ids=doc_ids)


def index_pdf(
    pdf_path: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None
) -> List[str]:
    """
    Pipeline entry point: loads & splits PDF using app/ingestion.py and indexes chunks in Chroma.

    Args:
        pdf_path: Path to the target PDF file.
        chunk_size: Character chunk size for RecursiveCharacterTextSplitter.
        chunk_overlap: Character overlap for RecursiveCharacterTextSplitter.
        persist_directory: Custom persistence directory path.
        collection_name: Custom Chroma collection name.
        embedding_function: Custom embedding instance.

    Returns:
        List of generated document IDs in Chroma.
    """
    chunks = load_and_split_pdf(
        file_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return index_documents(
        documents=chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )


def similarity_search(
    query: str,
    k: Optional[int] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None
) -> List[Document]:
    """
    Performs similarity search against Chroma vector store returning matching Documents.
    """
    top_k = k if k is not None else DEFAULT_TOP_K
    vector_store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return vector_store.similarity_search(query=query, k=top_k)


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_function: Optional[Embeddings] = None
) -> List[Tuple[Document, float]]:
    """
    Performs similarity search against Chroma vector store returning (Document, relevance_score) tuples.

    Args:
        query: User question or search query string.
        k: Top-k matching documents count (defaults to TOP_K env var, default 4).
        persist_directory: Custom persistence directory path.
        collection_name: Custom Chroma collection name.
        embedding_function: Custom embedding instance.

    Returns:
        List of (Document, float_score) tuples for Phase 3 relevance decision making.
    """
    top_k = k if k is not None else DEFAULT_TOP_K
    vector_store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    return vector_store.similarity_search_with_score(query=query, k=top_k)
