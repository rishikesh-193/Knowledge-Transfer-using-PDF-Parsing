import gc
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_ingestion import create_pdf_helper
from tests.test_vectorstore import DeterministicFakeEmbeddings
from tests.test_rag_chain import FakeChatModel
from app.rag import get_rag_chain, FALLBACK_RESPONSE
from server import create_app


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.db_dir = os.path.join(self.temp_dir, "test_api_vectorstore")
        self.fake_embeddings = DeterministicFakeEmbeddings(vector_dim=3072)
        self.fake_llm = FakeChatModel()

        # Build custom RAG chain with fake components for offline API unit testing
        self.rag_chain = get_rag_chain(
            llm=self.fake_llm,
            embedding_function=self.fake_embeddings,
            persist_directory=self.db_dir,
            collection_name="test_api_collection",
            relevance_threshold=1000.0
        )

        self.app = create_app(
            rag_chain_override=self.rag_chain,
            embedding_function_override=self.fake_embeddings,
            persist_directory_override=self.db_dir,
            collection_name_override="test_api_collection"
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_endpoint(self):
        """Test 1: GET /health returns 200 OK with status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_upload_valid_pdf(self):
        """Test 2: POST /upload with a valid PDF indexes chunks and returns 200 OK."""
        pdf_path = self.temp_path / "sample_kt.pdf"
        create_pdf_helper(pdf_path, ["Page 1 text content for API upload test."])

        with open(pdf_path, "rb") as f:
            files = {"file": ("sample_kt.pdf", f, "application/pdf")}
            response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["chunks_indexed"], 0)
        self.assertEqual(data["filename"], "sample_kt.pdf")

    def test_upload_invalid_file(self):
        """Test 3: POST /upload with a non-PDF file returns HTTP 400 Bad Request."""
        text_path = self.temp_path / "notes.txt"
        with open(text_path, "w") as f:
            f.write("Plain text file content")

        with open(text_path, "rb") as f:
            files = {"file": ("notes.txt", f, "text/plain")}
            response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Expected a '.pdf' file", response.json()["detail"])

    def test_upload_scanned_pdf(self):
        """Test 4: POST /upload with a scanned/image-only PDF returns HTTP 400 Bad Request."""
        scanned_pdf_path = self.temp_path / "scanned_doc.pdf"
        create_pdf_helper(scanned_pdf_path, [""])  # Empty text stream

        with open(scanned_pdf_path, "rb") as f:
            files = {"file": ("scanned_doc.pdf", f, "application/pdf")}
            response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 400)
        self.assertIn("contains no extractable text", response.json()["detail"])

    def test_rag_invoke_answerable_question(self):
        """Test 5 & 11: POST /rag/invoke returns structured answer and source metadata."""
        # Index document first
        pdf_path = self.temp_path / "architecture.pdf"
        create_pdf_helper(pdf_path, ["Microservices architecture with Redis caching layer."])

        with open(pdf_path, "rb") as f:
            self.client.post("/upload", files={"file": ("architecture.pdf", f, "application/pdf")})

        payload = {"input": {"question": "Microservices architecture Redis caching"}}
        response = self.client.post("/rag/invoke", json=payload)

        self.assertEqual(response.status_code, 200)
        output = response.json().get("output", {})
        self.assertIn("answer", output)
        self.assertIn("sources", output)
        self.assertNotEqual(output["answer"], FALLBACK_RESPONSE)
        self.assertEqual(len(output["sources"]), 1)
        self.assertEqual(output["sources"][0]["source"], "architecture.pdf")

    def test_rag_invoke_irrelevant_question(self):
        """Test 6 & 7: POST /rag/invoke with irrelevant query returns exact fallback sentence."""
        # App with strict relevance threshold
        strict_chain = get_rag_chain(
            llm=self.fake_llm,
            embedding_function=self.fake_embeddings,
            persist_directory=self.db_dir,
            collection_name="test_api_collection",
            relevance_threshold=0.00001
        )
        app = create_app(
            rag_chain_override=strict_chain,
            embedding_function_override=self.fake_embeddings,
            persist_directory_override=self.db_dir
        )
        client = TestClient(app)

        payload = {"input": {"question": "Unrelated topic quantum mechanics physics"}}
        response = client.post("/rag/invoke", json=payload)

        self.assertEqual(response.status_code, 200)
        output = response.json().get("output", {})
        self.assertEqual(output["answer"], FALLBACK_RESPONSE)
        self.assertEqual(output["sources"], [])

    def test_rag_schema_endpoints(self):
        """Test 8 & 9: LangServe schema endpoints return 200 OK."""
        input_schema_resp = self.client.get("/rag/input_schema")
        self.assertEqual(input_schema_resp.status_code, 200)

        output_schema_resp = self.client.get("/rag/output_schema")
        self.assertEqual(output_schema_resp.status_code, 200)

    def test_rag_playground_endpoint(self):
        """Test 10: GET /rag/playground/ returns HTTP 200 OK (UI loads)."""
        response = self.client.get("/rag/playground/")
        self.assertEqual(response.status_code, 200)

    def test_rag_stream_endpoint(self):
        """Test LangServe /rag/stream endpoint returns HTTP 200 streaming events."""
        payload = {"input": {"question": "What is the system architecture?"}}
        response = self.client.post("/rag/stream", json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
