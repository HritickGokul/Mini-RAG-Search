# 🪶 Mini RAG Search (Ollama-only)

A lightweight Retrieval-Augmented Generation (RAG) search engine that runs fully local.
It combines chunked retrieval with LLM-powered answers using Ollama
 and llama3:8b, making it perfect for niche, offline, or privacy-focused use cases.

No cloud dependencies. No API costs. Just local search + local answers.

## 🚀 Features

🔍 Chunked Retrieval: Splits documents into semantically meaningful chunks for accurate retrieval.

🤖 Local LLM (llama3:8b): Runs via Ollama for fast, private, and cost-free inference.

📚 Mini Search Engine: Query across your documents and get context-aware responses.

🎛️ Streamlit UI: Simple, interactive web interface for search and answer exploration.

🔒 Privacy-first: No external API calls. All processing happens locally on your machine.

## 📦 Installation

Clone this repository:

```bash

git clone https://github.com/yourusername/mini-rag-search.git
cd mini-rag-search

```
Create a virtual environment:

```bash
python -m venv .venv
```
Activate it:

1. Windows
   ```bash
   .venv\Scripts\activate
   ```
2. Mac/Linux
   ```bash
   source .venv/bin/activate
   ```
Install dependencies:

```bash
pip install -r requirements.txt
```
## ▶️ Usage

Run the app with Streamlit:

```bash
streamlit run app.py
```
Then open http://localhost:8501
 in your browser.

Upload documents (txt/markdown)

Enter your query

Mini RAG Search retrieves relevant chunks

## 🛠️ How It Works

1. Chunking

  * Documents are split into overlapping chunks (e.g., 512 tokens)
    
  * Ensures context continuity
    
  * Embedding & Retrieval
    
  * Each chunk is embedded into a vector space
    
  * Queries are embedded the same way
    
  * Chunks are ranked by semantic similarity

2. LLM Answering

  * Top chunks are passed to Ollama llama3:8b

3. Generates an answer grounded in retrieved context

  * Ollama (llama3:8b) generates a contextualized answer




