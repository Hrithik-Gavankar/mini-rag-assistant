# Mini RAG Assistant

A small **Retrieval-Augmented Generation (RAG)** demo for GenAI lectures: answer questions using `course_notes.txt` instead of the model’s general knowledge.

```
Documents → Chunking → Embeddings → Similarity search → LLM + context → Answer + sources
```

## What you need

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) (for answer generation)
- ~500MB disk for the first run (Sentence Transformers downloads a small embedding model)

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/mini-rag-assistant.git
cd mini-rag-assistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY

streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Try these questions

| Question | What to expect |
|----------|----------------|
| Explain the difference between RAG and fine-tuning based on our notes. | Grounded answer from section 3 |
| How does MCP help an AI application use external tools? | Grounded answer from section 5 |
| Which image-generation limitations did we discuss? | Grounded answer from section 6 |
| What is the exact submission deadline for the final project? | Should say the deadline is **not** in the notes |

Always check **Retrieved sources** — if the wrong chunks appear, the answer may be wrong even when the model sounds confident.

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `rag.py` | Chunking, embeddings, retrieval, LLM call |
| `course_notes.txt` | Sample lecture notes (replace with your own) |
| `.env` | API key (never commit this) |

## How it works (teaching version)

1. **Load** `course_notes.txt`
2. **Chunk** into ~200-word segments with overlap
3. **Embed** each chunk with Sentence Transformers (`all-MiniLM-L6-v2`)
4. **Retrieve** top 3 chunks by cosine similarity to the question
5. **Generate** with OpenAI using only those chunks as context

No vector database — fine for classroom-sized notes.

## Customize

- Edit `course_notes.txt` with your lecture content.
- Tune `CHUNK_WORDS`, `CHUNK_OVERLAP`, and `TOP_K` in `rag.py`.
- Change `OPENAI_MODEL` in `.env` if your account supports other models.

## License

MIT — use freely for teaching and labs.
