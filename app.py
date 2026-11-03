"""StudyMate web UI.   Run with:  streamlit run app.py"""

import streamlit as st

from studymate import config
from studymate.llm import LLMUnavailable, OllamaClient
from studymate.loaders import SUPPORTED
from studymate.rag import StudyMate
from studymate.store import KnowledgeBase

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and index...")
def get_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.index_folder()
    return kb


kb = get_kb()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📚 StudyMate")
    st.caption("Free, offline RAG over your engineering notes")

    model = st.text_input("Ollama model", value=config.LLM_MODEL)
    top_k = st.slider("Passages sent to the LLM", 1, 8, config.TOP_K)
    llm = OllamaClient(model)
    try:
        llm.check()
        st.success(f"LLM ready: {model}")
    except LLMUnavailable as exc:
        st.error(str(exc))

    st.divider()
    st.subheader("Your notes")
    uploads = st.file_uploader(
        "Add notes", type=[s.lstrip(".") for s in SUPPORTED], accept_multiple_files=True
    )
    if uploads and st.button("Add and index"):
        config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (config.NOTES_DIR / f.name).write_bytes(f.getbuffer())
        stats = kb.index_folder()
        st.success(f"Indexed {stats['chunks']} new chunks")
    if st.button("Re-scan notes folder"):
        stats = kb.index_folder()
        st.info(f"{stats['added']} new, {stats['updated']} updated, {stats['removed']} removed")
    for s in kb.sources():
        st.write(f"• {s}")
    st.caption(f"{kb.count()} chunks indexed")
    if st.button("Clear chat"):
        st.session_state.messages = []

# --------------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Ask your notes")
if kb.count() == 0:
    st.warning(f"No notes indexed. Add .md/.txt/.pdf files to {config.NOTES_DIR} or upload them.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(s)

question = st.chat_input("e.g. What are the four Coffman conditions?")
if question and kb.count() > 0:
    history = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
        for i in range(0, len(st.session_state.messages) - 1, 2)
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    app = StudyMate(kb, llm)
    with st.chat_message("assistant"):
        retrieval = app.retrieve(question, history, top_k=top_k)
        try:
            answer = st.write_stream(app.stream_answer(question, retrieval, history))
        except LLMUnavailable as exc:
            answer = f"⚠️ {exc}\n\nMost relevant passages from your notes are shown below."
            st.markdown(answer)
        sources = []
        if retrieval.hits and retrieval.in_scope:
            for i, h in enumerate(retrieval.hits, 1):
                d = "" if h.distance is None else f" — distance {h.distance:.2f}"
                sources.append(f"**[{i}] {h.label}**{d}\n\n> {h.text[:400]}...")
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(s)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
