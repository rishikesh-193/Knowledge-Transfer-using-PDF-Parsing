import os
import sys
import tempfile
from typing import Any, Optional

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel, Field

from app.ingestion import (
    load_and_split_pdf,
    PDFIngestionError,
    UnsupportedPDFError
)
from app.vectorstore import index_documents
from app.rag import get_rag_chain

load_dotenv()


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Server health status")


class UploadResponse(BaseModel):
    status: str = Field(..., description="Upload status string")
    chunks_indexed: int = Field(..., description="Total text chunks indexed into Chroma")
    pages: int = Field(..., description="Total pages parsed from PDF")
    filename: str = Field(..., description="Original filename of the uploaded PDF")


def create_app(
    rag_chain_override: Optional[Any] = None,
    embedding_function_override: Optional[Any] = None,
    persist_directory_override: Optional[str] = None,
    collection_name_override: Optional[str] = None
) -> FastAPI:
    """
    FastAPI application factory mounting LangServe routes and upload endpoints.
    Allows overriding RAG chain and embeddings for fast offline testing.
    """
    app = FastAPI(
        title="RAG Agent - KT",
        version="1.0",
        description="LangServe application for RAG Agent using PDF Parsing",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return HealthResponse(status="ok")

    # 2. PDF Upload endpoint
    @app.post("/upload", response_model=UploadResponse)
    async def upload_pdf(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Invalid file type. Expected a '.pdf' file.")

        # Save upload to temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        try:
            content = await file.read()
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(content)

            # Ingest & split using app/ingestion.py
            chunks = load_and_split_pdf(tmp_path)
            
            # Normalize source metadata to original uploaded filename
            pages_set = set()
            for chunk in chunks:
                chunk.metadata["source"] = file.filename
                if "page" in chunk.metadata:
                    pages_set.add(chunk.metadata["page"])

            pages_count = len(pages_set) if pages_set else 1

            # Index chunks in Chroma using app/vectorstore.py
            doc_ids = index_documents(
                documents=chunks,
                persist_directory=persist_directory_override,
                collection_name=collection_name_override,
                embedding_function=embedding_function_override
            )

            return UploadResponse(
                status="ok",
                chunks_indexed=len(doc_ids),
                pages=pages_count,
                filename=file.filename
            )

        except (UnsupportedPDFError, PDFIngestionError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # 3. Mount Phase 3 RAG Runnable via LangServe add_routes()
    chain = rag_chain_override if rag_chain_override is not None else get_rag_chain(
        embedding_function=embedding_function_override,
        persist_directory=persist_directory_override,
        collection_name=collection_name_override
    )

    add_routes(
        app,
        chain,
        path="/rag"
    )

    return app


# Default application instance for Uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
