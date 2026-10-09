"""Minimal RAG pipeline: chunk → embed → retrieve → generate."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40
TOP_K = 3


@dataclass
class Chunk:
    index: int
    text: str


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def load_document(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def chunk_text(text: str, words_per_chunk: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    words = re.split(r"\s+", text)
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        piece = " ".join(words[start:end])
        chunks.append(Chunk(index=idx, text=piece))
        idx += 1
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


class MiniRAG:
    def __init__(self, notes_path: str):
        self.notes_path = notes_path
        self._embedder = SentenceTransformer(EMBED_MODEL)
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        text = load_document(notes_path)
        self.chunks = chunk_text(text)
        if not self.chunks:
            raise ValueError(f"No content found in {notes_path}")

        texts = [c.text for c in self.chunks]
        self.embeddings = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[RetrievalResult]:
        q_vec = self._embedder.encode(question, convert_to_numpy=True, show_progress_bar=False)
        scores = _cosine_similarity(q_vec, self.embeddings)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in order
        ]

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        context_blocks = [f"[Source {h.chunk.index + 1}]\n{h.chunk.text}" for h in hits]
        context = "\n\n".join(context_blocks)

        system = (
            "You are a teaching assistant for a Generative AI course. "
            "Answer ONLY using the provided course notes. "
            "If the notes do not contain enough information, say clearly that "
            "you cannot find the answer in the course materials. "
            "Do not invent deadlines, grades, or policies."
        )
        user = (
            f"Course notes (retrieved excerpts):\n\n{context}\n\n"
            f"Student question: {question}\n\n"
            "Give a concise answer and mention which source numbers you used."
        )

        if not os.getenv("OPENAI_API_KEY"):
            return {
                "answer": (
                    "Set OPENAI_API_KEY in your .env file to generate answers. "
                    "Retrieval still works — see the sources below."
                ),
                "sources": hits,
                "used_llm": False,
            }

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""

        return {"answer": text.strip(), "sources": hits, "used_llm": True}
