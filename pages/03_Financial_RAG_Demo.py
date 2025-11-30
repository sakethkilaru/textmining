import os
import glob
import numpy as np
import streamlit as st
from openai import OpenAI

# ========== CONFIG ==========
DATA_DIR = "financial_statements"          # where your .txt files live
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"                 # change if you want

# ========== OPENAI CLIENT ==========
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        st.error(
            "OPENAI_API_KEY not set.\n\n"
            "Set it as an environment variable or add it to .streamlit/secrets.toml."
        )
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

# ========== TEXT SPLITTING & EMBEDDING ==========
def split_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    """Very simple character-based splitter."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts, model=EMBED_MODEL, batch_size=64):
    """Embed a list of texts using OpenAI embeddings; returns (N, D) array."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        embeddings.extend(batch_embeddings)
    return np.vstack(embeddings)


def cosine_sim(a, b):
    """Cosine similarity between (N,D) and (D,) -> (N,) sims."""
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


# ========== LOAD CORPUS & BUILD INDEX ==========
@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def load_corpus_and_index(data_dir=DATA_DIR, version="v2"):
    # just adding version="v2" changes the cache key
    docs = []
    metadatas = []

    txt_paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not txt_paths:
        st.error(f"No .txt files found in {data_dir}. Put your company txts there.")
        st.stop()

    for path in txt_paths:
        ticker = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text(text, chunk_size=1500, overlap=200)
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append(
                {
                    "ticker": ticker,
                    "chunk_id": i,
                    "source": path,
                }
            )

    embeddings = embed_texts(docs)
    return docs, metadatas, embeddings


def retrieve(query, docs, metadatas, embeddings, k=5):
    query_emb = embed_texts([query])[0]        # (D,)
    sims = cosine_sim(embeddings, query_emb)[:, 0]  # (N,)
    top_idx = np.argsort(sims)[::-1][:k]

    retrieved = []
    for idx in top_idx:
        retrieved.append(
            {
                "text": docs[idx],
                "meta": metadatas[idx],
                "score": float(sims[idx]),
            }
        )
    return retrieved


# ========== LLM ANSWERS ==========
def answer_no_rag(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful financial Q&A assistant. "
                "Answer using your own knowledge. "
                "If you are unsure, say you are not sure instead of making up numbers."
            ),
        },
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def answer_with_rag(question: str, retrieved_chunks) -> str:
    context_parts = []
    for item in retrieved_chunks:
        ticker = item["meta"]["ticker"]
        score = item["score"]
        context_parts.append(f"[{ticker} | score={score:.3f}]\n{item['text']}\n")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a careful financial Q&A assistant.\n"
        "You will be given context extracted from company financial statements.\n"
        "Use ONLY this context to answer the question.\n"
        "If the answer is not in the context, say you cannot answer from the provided documents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer clearly and include specific numbers only if they are present in the context."
    )

    messages = [
        {"role": "system", "content": "You answer financial questions using provided context only."},
        {"role": "user", "content": prompt},
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ========== PAGE UI ==========
def main():
    st.title("📚 Financial RAG Demo")
    st.write(
        "This page lets you compare **No-RAG vs RAG** answers over the 10 company "
        "financial text files stored in `financial_statements/`."
    )

    with st.sidebar:
        st.header("RAG Settings")
        top_k = st.slider("Number of chunks to retrieve (k)", 1, 10, 5)
        st.caption("Higher k = more context but longer prompts and more noise.")

    with st.spinner("Loading corpus and building vector index..."):
        docs, metadatas, embeddings = load_corpus_and_index()

    question = st.text_area(
        "Ask a financial question (e.g. `What was Tesla's revenue in 2023?`)",
        height=80,
    )

    if st.button("Run RAG vs No-RAG"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        retrieved = retrieve(question, docs, metadatas, embeddings, k=top_k)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("❌ No-RAG Answer")
            no_rag_answer = answer_no_rag(question)
            st.write(no_rag_answer)

        with col2:
            st.subheader("✅ RAG Answer")
            rag_answer = answer_with_rag(question, retrieved)
            st.write(rag_answer)

        st.markdown("---")
        st.subheader("Retrieved Context Chunks")
        for i, item in enumerate(retrieved, start=1):
            meta = item["meta"]
            with st.expander(
                f"Chunk {i}: {meta['ticker']} (score={item['score']:.3f}) - {meta['source']}"
            ):
                st.write(item["text"])


if __name__ == "__main__":
    main()
