"""Retrieval-augmented generation over the restaurant knowledge base."""

from __future__ import annotations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .logging_config import get_logger
from .prompts import PROMPT_TEMPLATE

logger = get_logger(__name__)


class RagService:
    """Loads a document, indexes it, and answers queries grounded in it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        ollama_kwargs: dict[str, str] = {"model": settings.ollama_model}
        if settings.ollama_base_url:
            ollama_kwargs["base_url"] = settings.ollama_base_url

        self._embeddings = OllamaEmbeddings(**ollama_kwargs)
        self._llm = OllamaLLM(**ollama_kwargs)
        self._vector_store = InMemoryVectorStore(self._embeddings)
        self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = self._prompt | self._llm

        self._index_knowledge_base()

    def _index_knowledge_base(self) -> None:
        path = self._settings.knowledge_base_path
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")

        logger.info("Indexing knowledge base from %s", path)
        documents = TextLoader(str(path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)
        self._vector_store.add_documents(chunks)
        logger.info("Indexed %d document chunk(s)", len(chunks))

    def _find_related_documents(self, query: str) -> list[Document]:
        return self._vector_store.similarity_search(query)

    def answer(self, user_query: str, chat_history: str) -> str:
        """Generate a grounded answer for ``user_query``."""
        context_docs = self._find_related_documents(user_query)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        response = self._chain.invoke(
            {
                "user_query": user_query,
                "document_context": context_text,
                "chat_history": chat_history,
            }
        )
        return response.strip()
