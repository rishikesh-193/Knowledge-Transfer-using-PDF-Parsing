import gc
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

load_dotenv()

from tests.test_ingestion import create_pdf_helper
from tests.test_vectorstore import DeterministicFakeEmbeddings
from app.vectorstore import index_pdf
from app.rag import (
    get_rag_chain,
    RAGPipeline,
    FALLBACK_RESPONSE
)


class FakeChatModel(BaseChatModel):
    """Fake chat model for deterministic unit testing without API quota consumption."""
    response_text: str = "This is a grounded answer from document context."
    call_count: int = 0
    should_fail: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("LLM service generation failed simulated error.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"


class TestRAGChain(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.db_dir = os.path.join(self.temp_dir, "test_rag_vectorstore")
        self.collection_name = "test_rag_collection"
        self.fake_embeddings = DeterministicFakeEmbeddings(vector_dim=3072)
        self.fake_llm = FakeChatModel()

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_rag_chain(self, llm=None, relevance_threshold=1.25):
        return get_rag_chain(
            llm=llm or self.fake_llm,
            embedding_function=self.fake_embeddings,
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            relevance_threshold=relevance_threshold
        )

    def test_answerable_question(self):
        """Test 1: Relevant document context generates grounded answer with source metadata."""
        pdf_path = self.temp_path / "kt_doc.pdf"
        page_texts = ["Postgres database is used for relational persistence in Knowledge Transfer pipeline."]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        chain = self._build_rag_chain(relevance_threshold=1000.0)  # High threshold to allow match
        response = chain.invoke({"question": "Postgres database relational persistence"})

        self.assertIn("answer", response)
        self.assertIn("sources", response)
        self.assertNotEqual(response["answer"], FALLBACK_RESPONSE)
        self.assertEqual(len(response["sources"]), 1)
        self.assertEqual(response["sources"][0]["source"], "kt_doc.pdf")
        self.assertEqual(response["sources"][0]["page"], 1)

    def test_unanswerable_question_and_relevance_gate(self):
        """Test 2: Irrelevant query is rejected by relevance gate; exact fallback returned; LLM NOT called."""
        pdf_path = self.temp_path / "kt_doc.pdf"
        page_texts = ["Postgres database is used for relational persistence."]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        # Set strict relevance threshold 0.0 (fake embeddings produce distance > 0 for different strings)
        fake_llm = FakeChatModel()
        chain = self._build_rag_chain(llm=fake_llm, relevance_threshold=0.00001)

        response = chain.invoke({"question": "Unrelated topic quantum mechanics physics"})

        self.assertEqual(response["answer"], FALLBACK_RESPONSE)
        self.assertEqual(response["sources"], [])
        self.assertEqual(fake_llm.call_count, 0, "LLM must NOT be called when relevance gate rejects query.")

    def test_multiple_retrieved_chunks(self):
        """Test 3: Multiple retrieved chunks format context correctly and deduplicate sources."""
        pdf_path = self.temp_path / "multi_page_kt.pdf"
        page_texts = [
            "Page 1: System architecture uses Microservices pattern.",
            "Page 2: Microservices communicate using gRPC protocol."
        ]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        chain = self._build_rag_chain(relevance_threshold=1000.0)
        response = chain.invoke("Microservices gRPC protocol architecture")

        self.assertEqual(len(response["sources"]), 2, "Sources from page 1 and page 2 should be preserved.")
        pages = [s["page"] for s in response["sources"]]
        self.assertIn(1, pages)
        self.assertIn(2, pages)

    def test_empty_retrieval(self):
        """Test 5: Empty Chroma vector store returns exact fallback response."""
        empty_db_dir = os.path.join(self.temp_dir, "empty_rag_store")
        chain = get_rag_chain(
            llm=self.fake_llm,
            embedding_function=self.fake_embeddings,
            persist_directory=empty_db_dir,
            collection_name="empty_collection"
        )

        response = chain.invoke({"question": "What is the system architecture?"})

        self.assertEqual(response["answer"], FALLBACK_RESPONSE)
        self.assertEqual(response["sources"], [])

    def test_input_validation(self):
        """Test 6: Empty or whitespace question returns fallback response cleanly."""
        chain = self._build_rag_chain()

        res1 = chain.invoke("")
        self.assertEqual(res1["answer"], FALLBACK_RESPONSE)
        self.assertEqual(res1["sources"], [])

        res2 = chain.invoke({"question": "   "})
        self.assertEqual(res2["answer"], FALLBACK_RESPONSE)

        res3 = chain.invoke({})
        self.assertEqual(res3["answer"], FALLBACK_RESPONSE)

    def test_llm_failure_handling(self):
        """Test 7: Exception during LLM generation raises clean application exception."""
        pdf_path = self.temp_path / "kt_doc.pdf"
        page_texts = ["API authentication uses OAuth2."]
        create_pdf_helper(pdf_path, page_texts)

        index_pdf(
            pdf_path=str(pdf_path),
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
            embedding_function=self.fake_embeddings
        )

        failing_llm = FakeChatModel(should_fail=True)
        chain = self._build_rag_chain(llm=failing_llm, relevance_threshold=1000.0)

        with self.assertRaises(RuntimeError) as ctx:
            chain.invoke("API authentication")

        self.assertIn("RAG LLM generation failed", str(ctx.exception))

    def test_lcel_runnable_interface(self):
        """Verify get_rag_chain returns a valid LCEL RunnableLambda."""
        chain = self._build_rag_chain()
        self.assertIsInstance(chain, RunnableLambda)


if __name__ == "__main__":
    unittest.main()
