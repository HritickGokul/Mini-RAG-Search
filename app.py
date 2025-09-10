import streamlit as st
import trafilatura
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
import httpx
import requests
#############################################################################################################
@st.cache_resource(show_spinner=False)

#FUNCTION FOR SPLITTING THE GIVEN TEXT INTO CHUNKS
def split_text(text: str, max_words: int = 800, overlap: int = 120):
    """
    Split long text into overlapping chunks.
    - max_words: target size of each chunk (by words)
    - overlap: words carried over from the end of one chunk to the start of the next
    """
    sentences = re.split(r"(?<=[.!?])\s+", text) #splits the extracted text into sentences wherever . or ? or ! is found
    chunks, buf, count = [], [], 0 #defining variables
    for sent in sentences:
        w = len(sent.split()) #Number of words in that sentence
        if count + w > max_words and buf: #if the number of words greater than the maximum limit and the current chunk is not empty
            # now im closing current chunk
            chunk = " ".join(buf).strip() #current chunk is a string of all sentences in buf excluding white spaces
            if chunk: #if there is a chunk
                chunks.append(chunk) #append the chunk to the chunks list
            # starting a new chunk with overlap tail
            tail = " ".join(chunk.split()[-overlap:]) if overlap > 0 else "" #new chunk starts from the overlap of previous chunk
            buf = [tail, sent] if tail else [sent] #current chunk is the tail of last chunk plus the current sentence
            count = len(" ".join(buf).split()) #count is the length of new chunk.
        else:
            buf.append(sent) #append the sentence to the current chunk
            count += w #increase the count with the number of words in the sentence
    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

#FUNCTION OF EMBEDDING
def get_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)


#FUNTION FOR GETTING RESPONSE FROM THE MODEL
def generate_answer_via_ollama(question: str, top_indices, model: str = "llama3:8b", host: str = "http://localhost:11434"):
    # Build numbered context from retrieved chunks
    ctx_lines = []
    for j, idx in enumerate(top_indices, start=1):
        chunk = st.session_state["chunks"][idx][:1100]
        src = st.session_state["sources"][idx]
        ctx_lines.append(f"[{j}] {chunk}\n(Source: {src})")
    context = "\n\n".join(ctx_lines)

    sys = (
        "You are a cautious research assistant. Answer ONLY using the provided context. "
        "If insufficient, say so and list sources. Cite like [1], [2]. Keep ≤8 sentences. "
        "Then add 'Quoted evidence:' with 1–2 short quotes."
    )
    user = f"Question: {question}\n\nContext:\n{context}"

    r = requests.post(
        f"{host}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "system", "content": sys},
                           {"role": "user", "content": user}]},
        timeout=120
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()



#############################################################################################################


st.title("Mini RAG Search App")
st.write("Hello, This is a mini RAG App I'm building")


with st.sidebar:
    st.subheader("Answering model")
    ollama_model = st.text_input("Ollama model", value="llama3:8b")
    st.caption("Run:  `ollama pull llama3:8b`  and ensure Ollama is running locally.")


#THIS IS WERE WE PASTE THE URLS
urls_text = st.text_area(
    "Paste 1–10 URLs (one per line):",
    value="https://en.wikipedia.org/wiki/Large_language_model\nhttps://huggingface.co/blog/introducing-datasets",
    height=120
)

#FETCH THE DETAILS FROM THE URL, CHUNK IT, EMBED IT AND STORE IT
if st.button("Fetch & Index URL"):
    all_chunks = []
    sources = []

    with st.spinner("Fetching and extracting..."):

        for raw in urls_text.splitlines():

            url = raw.strip()
            if not url:
                continue

            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                st.error(f"Could not fetch: {url}, moving onto the next URL")
                continue
            else:
                text = trafilatura.extract(
                    downloaded,
                    include_comments = False,
                    include_links = False
                ) or ""
                if not text.strip():
                    st.warning(f"Could not extract text: {url}, moving onto the next URL")
                else:
                    st.success("Done.")
                    st.write(f"**Character Count:** {len(text)}")

                    chunks = split_text(text, max_words = 800, overlap = 120)
                    all_chunks.extend(chunks)
                    sources.extend([url] * len(chunks))

    if not all_chunks:
        st.error("No chunks created from the provided URLs. Try different links.")
    else:
        # Embed chunks
        embedder = get_embedder()
        chunk_vecs = embedder.encode(all_chunks, normalize_embeddings=True)

        # Keep for search step
        st.session_state["chunks"] = all_chunks
        st.session_state["chunk_vecs"] = np.array(chunk_vecs)
        st.session_state["sources"] = sources



#FIND THE TOP MATCHES AND THEN USE THAT TO FEED IT TO THE MODEL TO GENERATE A RESPONSE
st.markdown("### Search your chunks")
query = st.text_input("Ask a question (e.g., 'What is RLHF?' or 'What is Constitutional AI?')")

if st.button("Search chunks"):
    if "chunk_vecs" not in st.session_state:
        st.warning("Ingest something first (fetch a URL) so I have chunks to search.")
    elif not query.strip():
        st.warning("Type a query first.")
    else:
        embedder = get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)[0]

        # cosine similarity = dot product because embeddings are normalized
        sims = st.session_state["chunk_vecs"] @ q_vec

        # top-5 indices, sorted by similarity
        top_k = 5
        top_idx = np.argsort(-sims)[:top_k]

        # st.subheader("Top matches")
        # for rank, idx in enumerate(top_idx, start=1):
        #     score = float(sims[idx])
        #     chunk_preview = st.session_state["chunks"][idx][:300]
        #     src = st.session_state["sources"][idx]
        #     st.markdown(f"**{rank}. similarity = {score:.3f}** \n_Source = {src}_")
        #     st.text(chunk_preview)

        st.markdown("---")
        st.subheader("Answer (local model via Ollama)")
        try:
            answer = generate_answer_via_ollama(query, top_idx, model=ollama_model)
            st.write(answer)
        except Exception as e:
            st.error(f"Ollama error: {e}")
            st.info("Is Ollama running and is the model pulled? Try:  `ollama pull llama3:8b`")



