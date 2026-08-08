import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from app.vectorstore import (
    similarity_search_with_score,
    DEFAULT_TOP_K
)

load_dotenv()

# Constants & Configuration
FALLBACK_RESPONSE = "This information is not available in the provided document."
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.25"))

# Grounding Prompt Template
GROUNDING_PROMPT_TEMPLATE = """You are a Knowledge Transfer assistant.
Answer ONLY using the provided document context below.
Do not use outside knowledge.

If the answer cannot be determined from the context, respond exactly:
"This information is not available in the provided document."

Context:
{context}

Question:
{question}"""

prompt_template = PromptTemplate(
    template=GROUNDING_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class SourceMetadata(BaseModel):
    page: int = Field(..., description="1-indexed page number from source PDF")
    source: str = Field(..., description="Filename of the source PDF document")


class RAGResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer text or fallback sentence")
    sources: List[SourceMetadata] = Field(default_factory=list, description="Source PDF page metadata")


class RAGQueryInput(BaseModel):
    question: str = Field(..., description="User query or question string")


class RAGPipeline:
    """
    RAG Answering Pipeline enforcing document grounding and score-based relevance gating.
    """
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ):
        self.llm = llm
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k or DEFAULT_TOP_K
        self.relevance_threshold = (
            relevance_threshold if relevance_threshold is not None else DEFAULT_RELEVANCE_THRESHOLD
        )

    def _get_llm(self) -> BaseLanguageModel:
        if self.llm is not None:
            return self.llm
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing or empty.")
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)

    def format_context(self, docs: List[Document]) -> str:
        """Formats a list of Document chunks into a single context string."""
        formatted_chunks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 1)
            formatted_chunks.append(f"[Document {idx} - Source: {source}, Page: {page}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

    def extract_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Deduplicates and extracts source/page metadata from retrieved documents."""
        seen = set()
        sources = []
        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            try:
                page = int(doc.metadata.get("page", 1))
            except (ValueError, TypeError):
                page = 1
            
            pair = (source, page)
            if pair not in seen:
                seen.add(pair)
                sources.append({"source": source, "page": page})
        return sources

    def invoke(self, input_data: Union[str, Dict[str, Any], RAGQueryInput]) -> Dict[str, Any]:
        """
        Executes the grounded RAG pipeline on a question input.

        Args:
            input_data: String question, dict with 'question' key, or RAGQueryInput instance.

        Returns:
            Dictionary matching RAGResponse schema: {'answer': str, 'sources': [{'source': str, 'page': int}]}
        """
        # 1. Parse & validate input question
        if isinstance(input_data, str):
            question = input_data
        elif isinstance(input_data, dict):
            question = input_data.get("question", "")
        elif hasattr(input_data, "question"):
            question = getattr(input_data, "question", "")
        else:
            question = str(input_data)

        question = question.strip() if question else ""
        if not question:
            return RAGResponse(answer=FALLBACK_RESPONSE, sources=[]).model_dump()

        # 2. Similarity retrieval with distance scores
        try:
            results_with_scores = similarity_search_with_score(
                query=question,
                k=self.top_k,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            # Handle store retrieval failures gracefully
            return RAGResponse(answer=FALLBACK_RESPONSE, sources=[]).model_dump()

        if not results_with_scores:
            return RAGResponse(answer=FALLBACK_RESPONSE, sources=[]).model_dump()

        # 3. Evaluate relevance using distance scores (Chroma L2 distance: lower = closer match)
        relevant_docs: List[Document] = []
        for doc, score in results_with_scores:
            if score <= self.relevance_threshold:
                relevant_docs.append(doc)

        if not relevant_docs:
            # Relevance gate rejected query - DO NOT call Gemini LLM API
            return RAGResponse(answer=FALLBACK_RESPONSE, sources=[]).model_dump()

        # 4. Format context & prompt
        context_str = self.format_context(relevant_docs)
        prompt_value = prompt_template.format(context=context_str, question=question)

        # 5. Invoke LLM for grounded answer generation
        try:
            llm_instance = self._get_llm()
            llm_response = llm_instance.invoke(prompt_value)
            answer_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            answer_text = answer_text.strip()
        except Exception as e:
            raise RuntimeError(f"RAG LLM generation failed: {e}") from e

        sources = self.extract_sources(relevant_docs)

        # 6. Verify fallback sentence consistency
        if not answer_text or FALLBACK_RESPONSE.lower() in answer_text.lower():
            return RAGResponse(answer=FALLBACK_RESPONSE, sources=[]).model_dump()

        return RAGResponse(answer=answer_text, sources=sources).model_dump()


def get_rag_chain(
    llm: Optional[BaseLanguageModel] = None,
    embedding_function: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    relevance_threshold: Optional[float] = None
) -> RunnableLambda:
    """
    Constructs a LangChain LCEL Runnable wrapping the grounded RAG pipeline.
    This Runnable can be directly mounted in Phase 4 using LangServe.
    """
    pipeline = RAGPipeline(
        llm=llm,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
        top_k=top_k,
        relevance_threshold=relevance_threshold
    )
    return RunnableLambda(pipeline.invoke)
