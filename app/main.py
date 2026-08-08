import sys
import io

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from langserve import add_routes
from app.chain import chain

app = FastAPI(
    title="RAG Agent - KT",
    version="1.0",
    description="LangServe application for RAG Agent using PDF Parsing",
)

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
