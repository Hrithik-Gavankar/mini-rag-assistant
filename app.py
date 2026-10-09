"""Streamlit chat UI for the mini RAG assistant."""

from pathlib import Path

import streamlit as st

from rag import MiniRAG

NOTES_PATH = Path(__file__).parent / "course_notes.txt"


@st.cache_resource
def get_rag() -> MiniRAG:
    return MiniRAG(str(NOTES_PATH))


def main() -> None:
    st.set_page_config(page_title="Mini RAG Assistant", page_icon="📚", layout="centered")
    st.title("Mini RAG Assistant")
    st.caption(
        "Ask questions about the GenAI lecture notes. Answers are grounded in "
        "`course_notes.txt` — retrieved chunks are shown so you can debug the pipeline."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. **Chunk** the notes\n"
            "2. **Embed** chunks locally (Sentence Transformers)\n"
            "3. **Retrieve** top matches with cosine similarity\n"
            "4. **Generate** an answer with an LLM + context"
        )
        st.divider()
        st.markdown("**Try these questions:**")
        st.markdown(
            "- RAG vs fine-tuning?\n"
            "- How does MCP help with tools?\n"
            "- Image-generation limitations?\n"
            "- Final project deadline? *(not in notes)*"
        )

    rag = get_rag()
    st.info(f"Loaded **{len(rag.chunks)}** chunks from `course_notes.txt`.")

    question = st.text_input("Your question", placeholder="e.g. Explain RAG vs fine-tuning from our notes")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving context and generating answer…"):
            result = rag.answer(question.strip())

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved sources")
        for hit in result["sources"]:
            with st.expander(f"Source {hit.chunk.index + 1} · similarity {hit.score:.3f}"):
                st.write(hit.chunk.text)


if __name__ == "__main__":
    main()
