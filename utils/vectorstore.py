"""
vectorstore.py
Builds and queries a FAISS vector store from document chunks.
Uses sentence-transformers (free, runs locally — no API cost).
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# Free, lightweight embedding model (~90MB, runs on CPU)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Load the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[str]) -> FAISS:
    """
    Convert text chunks into embeddings and store in FAISS index.
    FAISS = Facebook AI Similarity Search — blazing fast local vector DB.
    """
    embeddings = get_embeddings()
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def retrieve_relevant_chunks(vectorstore: FAISS, query: str, k: int = 8) -> str:
    """
    Retrieve the top-k most relevant chunks for a given query.
    Returns them as a single string to inject into the LLM prompt.
    """
    results = vectorstore.similarity_search(query, k=k)
    combined = "\n\n".join([f"[Chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(results)])
    return combined
