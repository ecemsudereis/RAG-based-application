
import streamlit as st
import os, pickle, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="LLM Course RAG Assistant", page_icon="📚", layout="wide")

@st.cache_resource
def load_system():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with open("rag_data.pkl", "rb") as f:
        data = pickle.load(f)
    index = faiss.deserialize_index(data["index_bytes"])
    return embed_model, index, data["chunks"]

embed_model, index, chunks = load_system()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a helpful teaching assistant for the "Introduction to Large Language Models" (SWE015) course at Istinye University. Answer the student's question using ONLY the provided context from the lecture slides. If the answer is not in the context, say "This topic is not covered in the provided lecture slides." You may use the previous conversation to understand follow-up questions (e.g. "can you explain it more simply?"). Be concise, clear, and educational. Always cite the slide name and page number you used."""

def retrieve(query, k=5):
    q_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    return [{"text": chunks[idx]["text"], "source": chunks[idx]["source"],
             "page": chunks[idx]["page"], "score": float(D[0][i])}
            for i, idx in enumerate(I[0])]

def ask(question, history, k=5):
    t0 = time.time()
    retrieved = retrieve(question, k=k)
    context = "\n\n---\n\n".join(
        [f"[Source: {r['source']}, page {r['page']}]\n{r['text']}" for r in retrieved])
    # Include last 4 turns of history for follow-up questions
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-8:]:  # last 4 user-assistant pairs
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user",
                     "content": f"Context from lecture slides:\n\n{context}\n\nQuestion: {question}"})
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages,
        temperature=0.2, max_tokens=600)
    return resp.choices[0].message.content, retrieved, time.time() - t0

# ===== UI =====
st.title("📚 Intro to LLM — RAG Assistant")
st.caption("Multi-turn chat over your lecture slides. Powered by FAISS + Llama 3.3 70B (Groq).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_log" not in st.session_state:
    st.session_state.sources_log = {}

with st.sidebar:
    st.header("System Info")
    st.write(f"**Total vectors:** {index.ntotal}")
    st.write(f"**Embedding dim:** 384")
    st.write("**Embedding:** all-MiniLM-L6-v2")
    st.write("**LLM:** llama-3.3-70b-versatile")
    st.write("**Vector store:** FAISS (IndexFlatL2)")
    st.write("**Memory:** last 4 turns")
    st.divider()
    st.subheader("Quick questions")
    examples = [
        "What is self-attention?",
        "How is it different from regular attention?",
        "Explain knowledge distillation.",
        "Why does distillation reduce model size?",
        "What is a vector database?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state["pending_q"] = ex
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_log = {}
        st.rerun()

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.sources_log:
            srcs = st.session_state.sources_log[i]
            with st.expander(f"Retrieved sources ({len(srcs)})"):
                for j, s in enumerate(srcs, 1):
                    st.markdown(f"**[{j}]** `{s['source']}` — page {s['page']} (score: {s['score']:.3f})")
                    st.caption(s["text"][:300] + "...")

# Handle pending question from sidebar button
pending = st.session_state.pop("pending_q", None)
user_input = st.chat_input("Ask a question about the lecture slides...")
if pending and not user_input:
    user_input = pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking..."):
            answer, sources, elapsed = ask(user_input, st.session_state.messages[:-1])
        st.markdown(answer)
        st.caption(f"Answered in {elapsed:.2f}s")
        msg_idx = len(st.session_state.messages)
        st.session_state.sources_log[msg_idx] = sources
        with st.expander(f"Retrieved sources ({len(sources)})"):
            for j, s in enumerate(sources, 1):
                st.markdown(f"**[{j}]** `{s['source']}` — page {s['page']} (score: {s['score']:.3f})")
                st.caption(s["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})
