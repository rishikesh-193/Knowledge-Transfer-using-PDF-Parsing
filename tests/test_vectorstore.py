import gc
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from tests.test_ingestion import create_pdf_helper
from app.vectorstore import (
    index_pdf,
    similarity_search,
    similarity_search_with_score,
    get_vector_store,
    DEFAULT_TOP_K
)


class DeterministicFakeEmbeddings(Embeddings):
    """
    Deterministic fake embeddings class for offline unit testing.
    Produces 3072-dimensional vector embeddings based on text hash, matching
    the exact dimension of models/gemini-embedding-001 without requiring network or API quota.
    """
    def __init__(self, vector_dim: int = 3072):
        self.vector_dim = vector_dim

    def _hash_text(self, text: str) -> list[float]:
        # Generate deterministic float vector of length vector_dim
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        return [float((h + i) % 1000) / 1000.0 for i in range(self.vector_dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_text(text)


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.db_dir = os.path.join(self.temp_dir, "test_vectorstore")
        self.collection_name = "test_kt_collection"
        self.fake_embeddings = DeterministicFakeEmbeddings(vector_dim=3072)

    def tearDown(self):
        gc.collect()  # Release SQLite handles on Windows
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_indexing_and_similarity_retrieval(self):
        """Test indexing PDF via app/ingestion.py pipeline and retrieving relevant chunks."""
        pdf_path = self.temp_path / "architecture_kt.pdf"
        page_texts = [
            "The database architecture uses Postgres for relational storage and Redis for fast caching.",
            "Authentication is handled via OAuth2 JWT tokens with automatic expiration after 24 hours."
        ]
        create_pdf_helper(pdf_path, page_texts)

        # 1. Index PDF into Chroma using offline deterministic embeddings
        doc_ids = index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        self.assertGreater(len(doc_ids), 0, "Document IDs should be generated upon indexing.")

        # 2. Similarity search for database question
        db_results = similarity_search(
            query="Postgres relational storage Redis caching",
            k=2,
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        self.assertGreater(len(db_results), 0, "Similarity search should return matching documents.")
        self.assertIn("Postgres", db_results[0].page_content)

    def test_similarity_search_with_score(self):
        """Test similarity_search_with_score returns (Document, float_score) tuples with metadata."""
        pdf_path = self.temp_path / "scored_doc.pdf"
        page_texts = ["Scanned PDF parsing with page metadata verification."]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        results_with_score = similarity_search_with_score(
            query="scanned PDF parsing",
            k=1,
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        self.assertEqual(len(results_with_score), 1)
        doc, score = results_with_score[0]
        self.assertIsInstance(score, float, "Similarity score must be a float.")
        self.assertEqual(doc.metadata.get("source"), "scored_doc.pdf")
        self.assertEqual(doc.metadata.get("page"), 1)

    def test_duplicate_indexing_prevention(self):
        """Test that re-indexing the same document updates entries instead of creating duplicate chunks."""
        pdf_path = self.temp_path / "duplicate_test.pdf"
        page_texts = ["Duplicate test page content for deterministic ID validation."]
        create_pdf_helper(pdf_path, page_texts)

        # Index once
        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        store1 = get_vector_store(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        initial_count = len(store1.get()["ids"])
        self.assertEqual(initial_count, 1)

        # Re-index same document
        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        store2 = get_vector_store(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        reindexed_count = len(store2.get()["ids"])
        self.assertEqual(reindexed_count, initial_count, "Re-indexing the same PDF must not create duplicate chunks.")

    def test_metadata_preservation(self):
        """Test that source filename and 1-indexed page number metadata are preserved on retrieved documents."""
        pdf_path = self.temp_path / "kt_guide.pdf"
        page_texts = [
            "Page 1: Overview of KT pipeline and document parsing engine.",
            "Page 2: Detailed setup instructions for deployment on Render cloud."
        ]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        results = similarity_search(
            query="Page 2 Render cloud deployment",
            k=1,
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        self.assertEqual(len(results), 1)
        top_doc = results[0]
        self.assertEqual(top_doc.metadata.get("source"), "kt_guide.pdf", "Metadata 'source' must match filename.")
        self.assertIn("page", top_doc.metadata, "Metadata 'page' must exist.")

    def test_top_k_behavior(self):
        """Test that top-k parameter limits retrieved results count correctly."""
        pdf_path = self.temp_path / "multi_page.pdf"
        page_texts = [
            f"Page {i}: System configuration section {i} details." for i in range(1, 6)
        ]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        # Test k=2
        res_k2 = similarity_search(
            query="System configuration details",
            k=2,
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        self.assertEqual(len(res_k2), 2, "Similarity search with k=2 must return exactly 2 documents.")

        # Test k=4 (default)
        res_default = similarity_search(
            query="System configuration details",
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )
        self.assertEqual(len(res_default), min(5, DEFAULT_TOP_K), f"Default search must return top {DEFAULT_TOP_K} documents.")

    def test_persistence_and_reloading(self):
        """Test that indexed data persists to disk and can be queried after re-instantiating Chroma store."""
        pdf_path = self.temp_path / "persistent_doc.pdf"
        page_texts = ["Persistent vector store verification test item."]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        # Re-load store from persist_directory cleanly
        reloaded_store = get_vector_store(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        reloaded_results = reloaded_store.similarity_search(query="verification test item", k=1)
        self.assertEqual(len(reloaded_results), 1, "Reloaded store must return indexed documents.")
        self.assertIn("Persistent vector store", reloaded_results[0].page_content)
        self.assertEqual(reloaded_results[0].metadata.get("source"), "persistent_doc.pdf")

    def test_empty_collection_behavior(self):
        """Test querying an empty Chroma collection returns an empty list without crashing."""
        empty_db_dir = os.path.join(self.temp_dir, "empty_vectorstore")
        results = similarity_search(
            query="Any query on empty vector store",
            k=4,
            persist_directory=empty_db_dir,
            collection_name="empty_collection",
            embedding_function=self.fake_embeddings
        )
        self.assertEqual(results, [], "Querying an empty collection must return an empty list.")


if __name__ == "__main__":
    unittest.main()
