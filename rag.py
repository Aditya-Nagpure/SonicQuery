import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic


_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Embed chunks and store in a FAISS index.

    Returns (index, chunks) — chunks kept in sync with index row order.
    """
    texts = [c["text"] for c in chunks]
    embeddings = _model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


def retrieve(query: str, index: faiss.Index, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Return the top_k most relevant chunks for a query.
    """
    query_vec = _model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer(query: str, retrieved: list[dict]) -> str:
    """
    Build a prompt from retrieved chunks and call Claude for an answer.
    """
    context = "\n\n".join(
        f"[{c['start']:.1f}s - {c['end']:.1f}s]: {c['text']}"
        for c in retrieved
    )

    prompt = f"""You are answering questions about a transcript.

Context:
{context}

Question: {query}

Answer concisely using only the context above."""

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text