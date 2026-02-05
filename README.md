---
title: Movie RAG Chat
emoji: 🎬
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.31.0"
app_file: src/app.py
pinned: false
---


# Movie Transcript RAG 🎬

A Retrieval-Augmented Generation (RAG) system for questioning movie transcripts.

## Features
- **Upload & Ingest**: Drag and drop `.srt` subtitle files.
- **Hybrid Search**: Combines Sentence Transformers (Vector) and BM25 (Keyword) search.
- **Scene-Aware Chunking**: Intelligently splits subtitles based on silence/time gaps.
- **LLM Answering**: Powered by Llama-3 (via Groq) for high-speed, accurate answers.
- **Timestamps**: Every answer cites the exact time range in the movie.

## Setup Locally
1. Clone the repo
2. `pip install -r requirements.txt`
3. Set `.env` keys (Groq & HuggingFace)
4. `streamlit run src/app.py`
